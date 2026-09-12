# unidecompiler 前端集合

一组针对 JS 反爬 / 设备指纹 / 风控 VM 的 [`unidecompiler`](https://github.com/Wker666/unidecompiler) 反编译前端，配套原始字节码样本。

## 前端 ↔ 目标算法

| 前端目录 | 目标签名 / 算法 | 输入扩展名 |
|----------|-----------------|------------|
| `qimeivm/` | **X-uskey**（腾讯设备指纹 / 风控签名） | `.qimei` |
| `bdvm/` | **a_bogus**（字节跳动 / 抖音系反爬签名） | `.bd` |
| `jdvm/` | **h5st**（京东 H5 端反爬签名） | `.jd` |
| `kasadavm/` | **Kasada VM**（Kasada 反爬虫 VM 保护逻辑） | `.kasada` |
| `mtgsigvm/` | **mtgsig**（美团 H5guard 设备指纹 / 签名） | `.mtgsig` |

## 各算法简介

- **X-uskey（qimeivm）**：腾讯的通用设备指纹与风控签名，qimei 是其设备指纹标识，VM 负责指纹 / 签名生成逻辑。
- **a_bogus（bdvm）**：字节跳动（抖音 / TikTok 系）的反爬签名参数，随请求动态生成，用于校验请求合法性。
- **h5st（jdvm）**：京东 H5 端反爬签名，配合风控做接口防护。
- **mtgsig（mtgsigvm）**：美团 H5guard SDK 的设备指纹与签名，用于反爬 / 风控。
- **Kasada（kasadavm）**：Kasada 反爬虫方案的核心 JS VM，业务逻辑经自定义 VM 保护。

## 字节码样本

原始字节码样本位于 `vm_files/`：

```
vm_files/
├── out.bd        # bdvm
├── out.qimei     # qimeivm
├── out.jd        # jdvm
├── out.kasada    # kasadavm
└── mtgsig/       # mtgsigvm（16 个独立 VM 容器）
    ├── vm_00_$_YIck.mtgsig
    ├── ...
    └── vm_15_$_JWQd.mtgsig
```

`mtgsig` 一个 SDK 内包含多个独立 VM 容器（`vm_00` … `vm_15`，对应不同的业务函数），故单独放在 `mtgsig/` 子目录。

## 目录结构

```
vm-frontend/
├── bdvm/         # a_bogus 前端
├── qimeivm/      # X-uskey 前端
├── jdvm/         # h5st 前端
├── kasadavm/     # Kasada 前端
├── mtgsigvm/     # mtgsig 前端
├── vm_files/     # 原始字节码样本
├── LICENSE       # PolyForm Noncommercial 1.0.0（禁止商用）
└── README.md
```
