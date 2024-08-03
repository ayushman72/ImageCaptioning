from tkinter import filedialog

file_path = filedialog.askopenfilename(title = "Select image",
                                       filetypes = (("jpeg files","*.jpg"),("png files",'*.png'),("all files","*.*")))

print(file_path)