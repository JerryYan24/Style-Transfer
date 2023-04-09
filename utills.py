import PIL.Image as Image
import torchvision.transforms as transforms


def load_img(img_path, img_size):
    """
    导入图片
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    return img

def show_img(img):
    """
    显示图片
    """
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()


def main():
    img_size = 512
    style_img_path = r"C:\Users\X\Downloads\pst\ref\1.jpg"
    content_img_path = r"C:\Users\X\Downloads\pst\ref\2.jpg"
    
    style_img = load_img(style_img_path, img_size)
    content_img = load_img(content_img_path, img_size)

    show_img(style_img)
    show_img(content_img)


if __name__=="__main__":
    main()