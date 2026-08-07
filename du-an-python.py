Tạo tài khoản cloud trên GITHUB Từ tài khoản mail outlook
Tạo folder dùng chung trên GITHUB
Link tài khoản
https://github.com/ha-dung-1085/du-an-dung-chung

Cài đặt ứng dụng Vs code, python, github → lưu ý: tích chọn và xác định Path lưu dữ liệu trong bước cài đặt
Cài đặt extension python, git, jupyter trong VS code

Tải folder dùng chung về máy tính/mac→ code
git clone https://github.com/username/TEN FOLDER.git

1. Mở thư mục làm việc
Vào đường dẫn đến folder dùng chung
CD "đường dẫn đến folder dùng chung"
Lưu ý: Cách Hiển thị Thanh đường dẫn (Path Bar) cố định ở đáy Finder trên Macbook
Giúp bạn luôn nhìn thấy vị trí của file ngay dưới đáy cửa sổ Finder:
Mở ứng dụng Finder.
Trên thanh menu góc trên màn hình, chọn Xem (View) 
--> chọn Hiển thị thanh đường dẫn (Show Path Bar) (hoặc nhấn tổ hợp phím Option + Command + P).
Nhìn xuống cạnh dưới cùng của cửa sổ Finder, bạn sẽ thấy đường dẫn thư mục đầy đủ. 
Muốn sao chép, chỉ cần nhấp chuột phải vào tên file/thư mục ở thanh này --> chọn Sao chép dưới dạng tên đường dẫn.

Cập nhật thay đổi lên GITHUB→code
git add .
hoặc
git add "tên file.đuôi mở rộng"
Tiếp theo:
git commit -m "thông tin thay đổi"
git push origin main

Cập nhật thay đổi từ GITHUB về máy tính/mac
git pull origin main


1. Cấu hình LFS 
git lfs install
git lfs track "*.csv"
git add .gitattributes
2. Git nhận diện lại các file lớn qua LFS
git add .
3. Commit và Push (Dùng lực mạnh)
git commit -m "Fix: Setup LFS and clean up large files"
git push origin main --force