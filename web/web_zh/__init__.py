"""中文版腦腫瘤分割網頁工作站 (Chinese-language web GUI).

這是 `web/` 套件的中文版本。它重用 `web` 套件裡所有的推論模組
（inference / preprocess / metrics_case / anatomy / risk / report / state），
只替換掉前端介面與摘要文字，並移除預測不確定性 (MC-Dropout uncertainty map)。

原版 `web/` 完全不受影響。

啟動方式：
    python -m uvicorn web_zh.server:app --host 0.0.0.0 --port 8001
"""
