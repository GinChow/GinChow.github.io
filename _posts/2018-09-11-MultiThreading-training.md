---
layout: post
title: tensorflow多线程训练的问题
author: Gin 
excerpt_separator: <!--more-->
tag: Python, Shell, Tensorflow
categories: [Python,Tech,Tensorflow]
---

在构建多线程队列训练机构时，发现多线程访问同一个generator时会引发generator already executing的exception，问题在于同一个generator在被多个线程访问的时候会发生冲突，所以需要使用threading lock来保证代码的线程安全性。
## Code sample1

```python
import threading  
  
class threadsafe_generator:  
    """Takes an generator and makes it thread-safe by 
    serializing call to the `next` method of given generator. 
    """  
    def __init__(self, gen):  
        self.gen = gen  
        self.lock = threading.Lock()  
  
    def __iter__(self):  
        return self.next()  
  
    def next(self):  
        with self.lock:  
            return self.gen.next()  
  
  
# 定义我们的生成器  
def all_keywords():  
    for row in csv.reader(open('companies.csv')):  
        if row[0]:  
           yield row[0]  
  
  
# 将其转换为线程安全的  
keywords = threadsafe_generator(all_keywords())  
# 然后在线程中就可以随意地使用keywords.next()而不必担心"generator already executing"异常了。  
```

## My code
```python
    # old version
    def _load_file_generator(self, fn_lists):
        for fn_in, fn_out, fn_len in fn_lists:
            data_in = np.load(os.path.join(self._input_src_dir,fn_in))
            data_out = np.load(os.path.join(self._out_src_dir, fn_out))
            data_len = np.load(os.path.join(self._len_src_dir, fn_len))
            yield data_in, data_out, data_len
    # new version
    def _load_new_generator(self):
        random.shuffle(self._train_list)
        count = 0
        while True:
            if count == self._train_list_len - 1:
                random.shuffle(self._train_list)
                count = 0
            fn_in, fn_out, fn_len  = self._train_list[count]
            count += 1
            yield fn_in, fn_out, fn_len
            
    def enqueue_training(self, sess, iterator, lock):
        stop = False
        while not stop:
            cur_time = time.time()
            with lock:
                fn_in, fn_out, fn_len = next(iterator)
            data_in = np.load(os.path.join(self._input_src_dir,fn_in))
            data_out = np.load(os.path.join(self._out_src_dir, fn_out))
            data_len = np.load(os.path.join(self._len_src_dir, fn_len))
            if self._coord.should_stop():
                stop = True
                break
            # cur_time = time.time()
            sess.run(self._train_enqueue_ops,
                     feed_dict={self._input_ele : data_in.reshape(-1, self._input_dim),
                                self._output_ele : data_out.reshape(-1, self._output_dim),
                                self._len_ele : data_len.reshape(1,)})
                # print("enqueue_time", time.time()-cur_time)

    # activate threads
    def start_enqueue_thread(self, sess):
        thread_list = []
        iterator = self._load_new_generator()
        lock = threading.Lock()
        for i in range(10):
            thread = threading.Thread(target=self.enqueue_training, args=(sess, iterator, lock))
            thread.daemon = True
            thread_list.append(thread)
        [thread.start() for thread in thread_list]
        return thread_list
```