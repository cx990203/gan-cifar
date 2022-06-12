# gan-cifar
使用基础的gan网络对图片进行生成</br>
使用cifar-10进行网络测试</br>
网络采用pytorch框架进行搭建
## 模型结构
gan网络采用生成网络和判别网络进行对抗</br>
生成网络输入直接为随机浮点数即可</br>
生成网络使用基本的nn.ConvTranspose2d进行反卷积操作，生成图片</br>
判别网络采用全卷积网络进行判别，最后得到一个浮点数，经过Sigmoid判别真假</br>
网络需要根据生成问题的不同而进行选择，生成网络与判别网络应该具有差不多的性能，才能达到比较好的效果，否则会导致一边过于强势，而使得另一边得不到任何学习。
## 生成结果
用于训练的图片（cifar-10）:</br>
![cifar-10](https://user-images.githubusercontent.com/77096562/173227813-a84ea9f3-1cc7-4153-9a96-1b3d58e5f3c0.png)</br>
生成结果:</br>
![epoch-242](https://user-images.githubusercontent.com/77096562/173230173-d5e47364-837b-4cba-a089-32eea5297477.png)
![epoch-243](https://user-images.githubusercontent.com/77096562/173227854-075265a9-fcdd-4106-b762-a6659e498f16.png)
![epoch-244](https://user-images.githubusercontent.com/77096562/173227857-d5d884a9-a55e-4c1f-8ef3-99b1b2aaf84c.png)
![epoch-245](https://user-images.githubusercontent.com/77096562/173227859-fdc26000-5f6f-45da-af2a-4fefa4c0df56.png)</br>
生成结果还是比较模糊的，且背景看起来相对比较花
## 模型损失
gan模型的损失比较特别，其损失不能收敛。虽然理想情况下损失应该是需要收敛的，但是实际训练难以达到理想效果</br>
一个比较理想的gan模型，其生成网络与判别网络的损失应该是上下浮动的</br>
![loss](https://user-images.githubusercontent.com/77096562/173227772-ef7ce2f4-b9e9-4f9c-b74a-4e25060a15d0.png)</br>
最后我做出来的结果显示，生成网络的损失从一开始就逐渐走高，判别网络的损失则是逐渐降低。最后两个损失均在一定范围上上下浮动，不断更新学习</br>
