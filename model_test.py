from torchinfo import summary
import torch
from Model_Training.Model_Def.TIM import ResNet, CorNet, DDCCor, models, DC, DCCR
from Model_Training.Model_Def.old_module import eaa
from Model_Training.Model_Def.EMBC import DesCor, CFNet, CF_Basic_l, CF_Basic_s
from ptflops import get_model_complexity_info

# 定义测试函数
def test(model_fn, input_channels=2, input_length=1250, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    x = torch.randn(32, input_channels, input_length).to(device)  # 将输入移动到指定设备
    # 实例化模型并移动到指定设备
    model = model_fn().to(device)
    
    print(f"Testing model: {model_fn.__name__}")
    print(model)

    try:
        summary(
            model,
            input_size=(32, input_channels, input_length),  # 输入尺寸
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=3,  # 控制显示的层级
            device=device,  # 指定设备
        )
    except Exception as e:
        print(f"Error in summarizing model: {e}\n")
    
    try:
        flops, params = get_model_complexity_info(
            model,
            (input_channels, input_length),  # 输入张量形状 (C, L)
            as_strings=True,                # 输出 FLOPs 和参数量为可读字符串
            print_per_layer_stat=True,      # 打印每一层的 FLOPs
            verbose=True,                   # 打印详细信息
        )
        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
    except Exception as e:
        print(f"Error in calculating FLOPs: {e}\n")

    # 执行前向传播
    try:
        output = model(x)
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!\n")
    except Exception as e:
        print(f"Error in forward pass: {e}\n")

# test(ResNet.Resnet18_1D)
# test(DCCR.DCCR_2_1)
test(CF_Basic_s.Resnet34_1D)
