# 环境

python3.6.5
requirements.txt

# 功能
densenet + bilstm + ctc实现ocr图片文字识别

# 训练
mytrain.bat

# 训练机器配置
处理器 i5-8250U CPU @ 1.60GHz 1.80 GHz
内存 16G
机器型号 Dell Latitude 5290
系统 win10

# 说明
贫穷的程序猿使用Dell Latitude 5290笔记本训练了6个月，整个训练过程可以在log.txt中观察到，还是很有意思的。

# 模型加载
下载：链接：https://pan.baidu.com/s/1ctxIBjHL01pwr2Dp6LuRjg 提取码：daky
解压后覆盖densenet_bilstm_ctc_ocr/ckpt

# 数据集
下载：链接：https://pan.baidu.com/s/1Hg872n7E01XbF9qQEtY4Jg 提取码：gflu
解压后覆盖densenet_bilstm_ctc_ocr/Synthetic_Chinese_String_Dataset


# log.txt预览
Write model to: D:\tensorflow\densenet_bilstm_ctc\ckpt\ocr_0.ckpt
Epoch 0 Step 000100, model loss 65.3909, LR: 0.0000100000
真实数据: 给我读某些章。忍不住
结果显示: 经的，的
Epoch 0 Step 000200, model loss 61.1985, LR: 0.0000100000
真实数据: 在五光十色的包装下面
结果显示: 在的在的
Epoch 0 Step 000300, model loss 74.0925, LR: 0.0000100000
真实数据: 黍。住宿费1200元
结果显示: 在，1人
Epoch 0 Step 000400, model loss 56.5987, LR: 0.0000100000
真实数据: 0年签署的现行合作协
结果显示: 0的是的
Epoch 0 Step 000500, model loss 63.8273, LR: 0.0000100000
真实数据: 有了自己的姓名CN域
结果显示: 了，一，的
Write model to: D:\tensorflow\densenet_bilstm_ctc\ckpt\ocr_0.ckpt
Epoch 0 Step 000600, model loss 65.5359, LR: 0.0000100000
真实数据: 益，又犯下第二个错误
结果显示: 在，、，、”
.........
.........
真实数据: 女者，科推动后勤改革
结果显示: 女者，科推动后勤改革
Epoch 0 Step 001300, model loss 0.0027, LR: 0.0000100000
真实数据: 从成交市场看，据日本
结果显示: 从成交市场看，据日本
Epoch 0 Step 001400, model loss 0.1308, LR: 0.0000100000
真实数据: 盛先生与店内数人员再
结果显示: 盛先生与店内数人员再
Epoch 0 Step 001500, model loss 0.1152, LR: 0.0000100000
真实数据: 过去所说的“世界上只
结果显示: 过去所说的“世界上只
Write model to: D:\tensorflow\densenet_bilstm_ctc\ckpt\ocr_0.ckpt
Epoch 0 Step 001600, model loss 7.9743, LR: 0.0000100000
真实数据: 涤，麦克-毕比得到1
结果显示: 溯，麦克-毕比得到1
Epoch 0 Step 001700, model loss 0.0244, LR: 0.0000100000
真实数据: 回白抚公，speci
结果显示: 回白抚公，speci

mail:illool@163.com
WX:illool