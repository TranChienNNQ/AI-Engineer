# Bản ghi lượt đo minh họa bài 40

[← Bài chính](README.md) · [Phụ lục về cách đọc benchmark](deep-dive.md)

Ngày thực hiện: **07/11/2026**. Đây là bản ghi một lượt, không là mốc hiệu năng phải đạt hoặc golden timing test. Số liệu trong bảng bài chính và hình 06 lấy từ lượt này.

Môi trường: python=3.14.6, numpy=2.5.4, platform=arm64, system=Darwin, accelerator=CPU only.

Compiler: `Apple clang version 21.0.0 (clang-2100.4.1.1)`.
Flags: `-O2 -std=c11 -ffp-contract=off -fno-fast-math`.
SHA256 source kernels.c: `8161412eb47538d39de7eab4356d1614ffa454e8a818ee8445b18f95f8f76301`.
Compiler subprocess: 2715354209ns; không bao gồm toàn bộ cold-start ứng dụng.

warm repeated CPU buffers; Python loop + ctypes dispatch + C compute; excludes validation, allocation and compilation; one native thread, CPU affinity/frequency not controlled; no HBM traffic measurement.

Đơn vị toàn bộ bảng/samples dưới đây là **ns/call**, mỗi sample là một block average. Quantile tuyến tính NumPy; ratio là hai median chia nhau. Input seed400, a=1.25,b=-0.5, các buffers FP32 preallocated và rời nhau.

## N=256

Inner repeats=200; 5 warm-up call/mode sau correctness call; 9 block/mode. Max error=0.0.

| Mode | p10 | Median | p90 |
|---|---:|---:|---:|
| separate | 997.791000 | 1028.330000 | 1072.461000 |
| fused | 715.039000 | 715.835000 | 752.126000 |

Ratio separate/fused: 1.436546131.

```text
separate: 1028.330000, 1061.670000, 1039.375000, 999.165000, 1011.460000, 998.125000, 996.455000, 1115.625000, 1038.540000
fused: 744.585000, 782.290000, 715.835000, 715.210000, 715.420000, 715.205000, 716.040000, 714.375000, 743.540000
```

Thứ tự mỗi sample: separate→fused ở sample0/2/4/6/8, fused→separate ở sample1/3/5/7.

## N=65536

Inner repeats=30; 5 warm-up call/mode sau correctness call; 9 block/mode. Max error=0.0.

| Mode | p10 | Median | p90 |
|---|---:|---:|---:|
| separate | 15381.380000 | 15488.900000 | 16206.926667 |
| fused | 5135.546667 | 5234.733333 | 5378.900000 |

Ratio separate/fused: 2.958870875.

```text
separate: 18795.833333, 15515.300000, 15498.633333, 15379.166667, 15488.900000, 15388.900000, 15559.700000, 15381.933333, 15430.566667
fused: 5505.566667, 5347.233333, 5152.800000, 5136.100000, 5218.033333, 5133.333333, 5261.100000, 5300.000000, 5234.733333
```

Thứ tự mỗi sample: separate→fused ở sample0/2/4/6/8, fused→separate ở sample1/3/5/7.

## N=1048576

Inner repeats=5; 5 warm-up call/mode sau correctness call; 9 block/mode. Max error=0.0.

| Mode | p10 | Median | p90 |
|---|---:|---:|---:|
| separate | 295186.560000 | 306008.200000 | 317030.040000 |
| fused | 82749.960000 | 90008.200000 | 92565.000000 |

Ratio separate/fused: 3.399781353.

```text
separate: 287800.000000, 306691.600000, 316608.400000, 297033.200000, 297825.000000, 306008.200000, 302708.200000, 313966.600000, 318716.600000
fused: 83466.600000, 79883.400000, 89541.600000, 84166.600000, 90008.200000, 90775.000000, 92050.000000, 94625.000000, 91241.600000
```

Thứ tự mỗi sample: separate→fused ở sample0/2/4/6/8, fused→separate ở sample1/3/5/7.

Chưa pin CPU/frequency, không đo DRAM/HBM counter, không GPU và không latency request. Các lần chạy lại có thể khác timing. Code correctness có thể đối chiếu; không assert các raw sample phải trùng.
