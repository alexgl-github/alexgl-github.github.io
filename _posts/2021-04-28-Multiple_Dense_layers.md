---
layout: post
mathjax: true
title:  "Dense layer with backpropagation in C++, part 2"
date:   2021-04-28 00:00:00 +0000
categories: github jekyll
---

## Dense layer with backpropagation in C++, part 2

Let's implement two Dense layer neural-network in C++.
We'll use C++ example from the [previous post]  (https://alexgl-github.github.io/github/jekyll/2021/04/16/Dense_layer.html),

2We'll use C++ example from the [previous post]  ({2021-04-28-Multiple_Dense_layers.md}),

"Dense layer with backpropagation in C++"

{{ site.baseurl }}{% link _posts/2021-04-28-Multiple_Dense_layers.md %}

{% post_url 2021-04-16-Dense_layer %}

changing Dense backpropagation function to eturn gradients used
as input to previous layer.

[previous_post]:  {2021-04-28-Multiple_Dense_layers.md}
