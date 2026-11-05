# Bài tập — Vòng train/eval và checkpoint

[Về bài học](README.md) · [Lời giải](solutions.md)

Tính tay hoặc dự đoán trước khi chạy. Dữ liệu trong đề là giả định, trừ câu hỏi về output lab.

## Bài 1. Một update có gì?

Với hai mẫu (0,1), (1,3), ŷ=wx+b, w=b=0 và mean MSE, tính loss, gradient và một bước SGD lr=0.1. Đo lại MSE trên cùng hai mẫu sau bước đó. Vì sao số đầu và số sau khác nhau? Đó có phải validation độc lập không?

## Bài 2. Batch cuối nhỏ

Ba squared error là 1, 9, 25. Chia thành batch hai mẫu và một mẫu. Tính mean hai batch mean và MSE trên mẫu. Viết cách cộng dồn đúng; điều gì đổi nếu mỗi mẫu có nhiều target?

## Bài 3. Mode không phải gradient

Dropout p=0.5 nhận vector toàn 1. Có thể nhận giá trị gì trong train? Eval trả gì? `no_grad()` có tắt Dropout không? Chạy mode modes và giải thích output. Vì sao đầu train_epoch phải gọi lại train sau evaluate?

## Bài 4. Early stopping tính tay

Val MSE lần lượt 0.800, 0.600, 0.595, 0.610, 0.580. Với patience=2, min_delta=0.010 và quy tắc strict trong bài, điền best epoch, bad_epochs và điểm dừng. Epoch 5 có được đọc không? Nếu min_delta=0 thì kết quả đổi ra sao?

## Bài 5. Chọn run không dùng test

Hai run có best val A=0.20, B=0.18. Một đồng nghiệp chấm test cả hai rồi chọn A vì test A=0.17, B=0.22. Sai ở đâu? Theo giao thức bài, chọn ai và lúc nào chấm test? Vì sao vẫn không kết luận B tốt hơn trên mọi dữ liệu?

## Bài 6. Epoch, step và log

48 mẫu train, batch size 16, 60 epoch không early stopping: bao nhiêu update? Validation 21 mẫu chia bao nhiêu batch? Vì sao train_online_mse không phải train_eval_mse? Có nhất thiết hai đường có một thứ tự cố định không?

## Bài 7. Best snapshot

Đoạn `best_state = model.state_dict()` rồi tiếp tục optimizer.step có bảo toàn model best không? Sửa nó. Best ở epoch 8, last ở epoch 11: file nào dùng dự đoán model đã chọn, file nào để tiếp tục tiến trình? Có thể ghép weight epoch 8 với optimizer epoch 11 rồi gọi là resume chính xác không?

## Bài 8. Momentum tính tay

SGD momentum μ=0.9, lr=0.1, w₀=1, g₁=2, g₂=1, không dampening/weight decay/Nesterov. Tính buffer và weight hai bước. Nếu ngắt sau bước đầu, chỉ tải w rồi tạo optimizer mới, w₂ là bao nhiêu? Chạy mode resume và xem chênh lệch parameter khi cố tình bỏ optimizer state.

## Bài 9. Khôi phục và seed

Job dừng sau epoch 5. Liệt kê trạng thái cần để resume giống lab. Vì sao chỉ gọi manual_seed lúc khởi động lại không đủ cho Dropout? Vì sao lab không phải lưu con trỏ sampler? Nếu dùng random augmentation từ NumPy hoặc resume giữa epoch, điều gì còn thiếu?

## Bài 10. Đọc artifact và viết báo cáo

Chạy mode study ở thư mục output mới. Đọc config.json, split.json, metrics.csv, summary.json, study.json. Viết 6–10 câu nêu dữ liệu, split, mô hình, hai cấu hình, quy tắc chọn, epoch best, điểm test và giới hạn. Phân biệt stopped=False ở epoch 60 với stopped=True ở epoch 30. Cần giữ những gì ngoài file weight để người khác kiểm tra báo cáo?

## Thực hành mở rộng

1. Chạy `selection_trace` với min_delta=0 và chuỗi ở bài 4. Đối chiếu dự đoán.
2. Dùng Config(batch_size=20, epochs=2, patience=10) trong thư mục tạm mới: dự đoán số step là 6 và các kích thước batch train là 20,20,8. Kiểm bằng DataLoader và summary.
3. Kiểm `evaluate` không sửa bất kỳ parameter nào bằng snapshot trước/sau; đổi batch size validation từ 16 thành 7 và kiểm MSE bằng nhau trong dung sai float64.
4. Chạy part và continued, cố ý thay Config.lr ở resume để quan sát lỗi cấu hình khác. Không dùng test để lựa chọn trong các thử nghiệm này.
