# WeChat Jump Helper

微信小程序《跳一跳》自动跳跃助手。

基于 **Python + OpenCV + ADB**，程序会自动识别人物和下一个方块，并计算按压时间完成跳跃。

## 现在支持什么？

目前支持：

- Android 手机
- macOS / Windows / Linux 电脑
- USB 调试或 Android 无线调试
- 自动识别不同竖屏分辨率

暂时不支持：

- iPhone
- 完全不使用电脑、只靠一台手机运行

## 新手怎么用？

### 1. 安装 Python

建议安装 Python 3.9 或更高版本。

### 2. 安装 ADB

安装 Android Platform Tools，并确保终端里能运行：

```bash
adb devices
```

### 3. 安装 Python 依赖

进入本项目文件夹后运行：

macOS / Linux：

```bash
python3 -m pip install -r requirements.txt
```

Windows：

```bash
python -m pip install -r requirements.txt
```

### 4. 连接 Android 手机

打开：

**设置 → 开发者选项 → USB 调试 / 无线调试**

连接成功后运行：

```bash
adb devices
```

看到手机后就可以继续。

### 5. 打开《跳一跳》

先在微信里进入《跳一跳》，让人物站在第一个方块上。

### 6. 启动程序

macOS / Linux：

```bash
python3 jump.py
```

Windows：

```bash
python jump.py
```

程序会自动开始跳。

想停止时，在终端按：

```text
Control + C
```

## 运行数据在哪里？

程序运行数据会保存在：

```text
~/.jump_helper/
```

程序结束后，当前文件夹还会生成：

```text
jump_all.csv
```

这是测试日志，普通使用者可以不管。

## 如果运行不了

先检查这三件事：

1. `adb devices` 能不能看到手机
2. 手机是不是竖屏
3. 人物是不是已经站在《跳一跳》的方块上

如果还是不行，可以到 GitHub 的 **Issues** 页面反馈，并附上：

- 手机型号
- Android 版本
- 报错截图
- 出问题时的游戏截图

## 说明

当前跳跃参数主要是在 1080×2400 的 Android 手机上测试出来的。

V5.2 已经加入分辨率适配，但不同手机、系统缩放或微信版本仍可能有差异。

这是一个学习和实验项目，不保证所有设备都能直接运行。
