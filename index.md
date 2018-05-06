---
layout: blog
title: blogs
---
<h1>This is my first blog site</h1>
<p>Hello world!</p>

<p>My post list:</p>
<ul>
	{% for post in site.posts %}
		<li><a href="{{site.baseurl}}{{post.url}}">{{post.title}}</a></li>
	{% endfor %}
</ul>