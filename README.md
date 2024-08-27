# Pilot Crop Monitoring
Hướng dẫn chạy thử nghiệm trên local

**Bước 1:** Clone git `git clone https://github.com/NhatUET/PilotCropMonitoring.git`

**Bước 2:** Tạo branch  
- Tạo branch mới: `git checkout -b <tên branch>`
- Update branch: `git push -u origin <tên branch>`
  
**Bước 3:** Cài VirtualEnv 
- Vào cmd của thư mục chứa mã nguồn
- Chạy lệnh `pip install virtualenv`
- Tiếp theo tạo môi trường env `python -m venv streamlit-env`
- Truy cập môi trường env `streamlit-env\env\Scripts\activate`

**Bước 4:** Cài đặt các thư viện cần dùng: `pip install -r requirements.txt` vào môi trường sản phẩm env. Nếu gặp lỗi 'pkg_resources' thì chạy lệnh `pip install --upgrade setuptools` trước.

**Bước 5:** Chạy ứng dụng thử nghiệm trên local: `streamlit run home.py`

 Hoặc có thể chạy thử nghiệm để người dùng trong cùng một mạng có thể truy cập:
 - Đầu tiên mở port trên máy và để port thường là 8080 (Có thể tùy chỉnh)
 - Sau đó chạy ứng dụng: `streamlit run home.py --server.port 8080`
 - Check ip: Bấm phím Windows, sau đó gõ cmd và check địa chỉ ip `ipconfig` (Window) hoặc `ifconfig` (Ubuntu). Sau đó tìm dòng có chứa `IPv4 Address`
 - Người dùng cùng mạng có thể truy cập: `<địa chỉ ipv4>:port` để xem giao diện ứng dụng streamlit


# Lưu ý:
- Phải tạo check branch trước khi pull hoặc push code `git checkout <tên branch>`

# Git Flow trong Project
**Mỗi khi bắt đầu code**

- A: đã code 1 số thay đổi so mới main --> `git stash --include-untracked`

- A chạy: `git checkout main` --> `git pull` --> `git checkout br_a` --> `git rebase main` --> `git stash pop`

**Mỗi khi commit**
 
- A: đang ở branch br_a: `git add . ` --> `git commit -m "<Nội dung commit>"` --> `git push origin br_a`

Chi tiết trong Git Flow: https://docs.google.com/document/d/1J71RQ6Uii1tgtLJzcwJcWXa28F_ONG1xYVL9rtqbJIg/edit?usp=sharing
