import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utills, PNST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 256
max_step = 50
style_weight = 1e5
content_weight = 1

def main():
    style_img_path = r"C:\Users\X\Downloads\pst\ref\1.jpg"
    content_img_path = r"C:\Users\X\Downloads\pst\ref\2.jpg"
    
    style_img = utills.load_img(style_img_path, img_size).to(device)
    content_img = utills.load_img(content_img_path, img_size).to(device)

    input_img = content_img.clone().to(device)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    cnn = PNST.VGG(style_img, content_img).to(device)

    step = 0
    while step < max_step:

        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        
        out, stylelosses, contentlosses = cnn.forward(input_img)
        loss = 0
        for sl in stylelosses: loss += sl.loss * style_weight
        for cl in contentlosses: loss += cl.loss * content_weight

        loss.backward()

        def closure():
            return loss

        optimizer.step(closure)

        print(step, "--", loss)
        step += 1
        input_img.data.clamp_(0, 1)


    print("end")
    utills.show_img(content_img)
    utills.show_img(input_img)

if __name__=="__main__":
    main()