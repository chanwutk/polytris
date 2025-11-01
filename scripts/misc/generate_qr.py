import qrcode

url = "https://github.com/chanwutk/polytris"

qr = qrcode.QRCode(version=1, box_size=10, border=0)
qr.add_data(url)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")

with open("qr.png", "wb") as f:
    img.save(f)