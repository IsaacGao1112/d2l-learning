import torch

x = torch.arange(4,dtype=torch.float32)
x.requires_grad_(True)
# print(x.grad)
y = 2*torch.dot(x,x)
# 这一步y的输出是 tensor(28., grad_fn=<MulBackward0>) 而非传统的tensor(28.)
# 原因就在于x使能了requires_grad_,
# print(x,'\n',y)
# print(x.grad)
y.backward()
# print(x.grad)# 这里是利用y.backward()才使得x.grad从None变成有值的
#这个应该就是所谓的反向传播，x本身只是自变量，没有梯度概念，但是y定义了函数后，就可以计算梯度。
#然后y向x反向传播，所以x的grad属性就有了值

# 注意这里自变量应该这么理解，x本身是自变量，他的维度就代表了自变量的维度（无论是矩阵还是向量还是单独的参数）
# 但是x在程序里又是被赋了值的，所以每当y反向传播了之后，x是会有具体的梯度值的。

# 每传播一次都会累计一次梯度，所以每次重新运算的时候需要zero一次
x.grad.zero_()
y = x.sum()
y.backward()
# print(x.grad)

# 2.5.2 非标量变量的反向传播（即函数f(x)出来的y是矩阵）
# TODO 这里为什么只有y.sum().backward()能出来结果的原因未知
x.grad.zero_()
y=x*x
y.sum().backward()
# print(x,'\n',x.grad)

# 2.5.3 分离计算
# 就是说z = y*x 而y=x*x
# 但是计算z关于x的梯度的时候，想
x.grad.zero_()
y = x*x
u = y.detach()
z = u*x
z.sum().backward()
# print(x.grad==u)

# 2.5.4 Python控制流的梯度计算

def f(a):
    # 这个函数本质上就是一个f(a)=k*a 但是具体这个比值k是多少由a差1000多少倍和正负来决定
    b = a*2
    
    while b.norm()<1000:# norm计算的是2范数，这里因为只有一个元素，有两个作用 1.化成标量 2.取绝对值
        b = b*2 #不断翻倍，直到b大于1000，while循环结束。
    
    if b.sum()>0: # 虽然2范数有绝对值，但这里是元素取和，相当于还是元素原值，具有正负
        c = b
        # print("positive condition! ",b,c)
    else:
        c = 100*b # 说明如果b为负，则再把b倍增100倍输出
        # print("negative condition!",b,c)
    return c
# size=()代表产生一个没有矩阵大小的1个元素
# a = torch.randn(size=(),requires_grad=True)
# 如果直接写1则会产生一个只有一个元素的一维数组
# a = torch.randn(1,requires_grad=True)
a = torch.randn((1),requires_grad=True)#加了括号也是一样，但是结果都是对的
# 但因为函数f约束，必须是一维的
# a = torch.randn((1,2),requires_grad=True)# 这个不行
# print(a)
# print(a.norm())
d = f(a)
d.backward()
# 因为f(a)是个比例函数，所以梯度就是比值。
# 用输出除以输入计算出的比例是真实比值，和反向传播出的梯度比较是否相等即可验证。
# print(a.grad == d/a) 

# 总结：所以当设定函数时，多个变量里面，如果需要求哪一维的偏导数，就在变量定义时对该维度requires_grad
# 当然如果a是一个向量，可以直接对向量requires_grad就可以得到该向量里所有元素维度的梯度了。
# 然后当函数计算完毕时，由因变量进行反向传播求得梯度。



