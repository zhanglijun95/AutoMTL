# MTL

* Multi-task learning

* We want to apply knowledge we learn from other tasks and apply them into new tasks
  * baby first recognize faces, and then apply this for recognize other objects
  * ML point of view: MTL introduce inductive bias that **prefer** hypothesis that explain **more than one** task.

* Hard parameter sharing: shared layer for all task, but keep Task-specific layers.
  * It reduce **overfitting**, which is normally due to too many unnecessary parameters

多个深度学习 减少一些参数

* programming framework，用户输入一个backbone
* 自动发现一个MTL，减少参数量。



* 需要了解pytorch API
* 基于pytorch API
* 深度学习框架
* 了解paper
* 编程思想