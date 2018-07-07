from PIL import Image, ImageDraw, ImageFont
import numpy as np
import string
import os
import pickle

alphabet = list(string.printable[:-6])

npy_file = []
fonts = []
char = []
bold = []
italics = []
for letter in alphabet:
    for font_file in os.listdir("../data/fonts"):
        try:#for desktopfile
            #choose font
            font = ImageFont.truetype('../data/fonts/'+font_file, 40, encoding ="unic")
            positions = [(5,-7),(25,-7),(5,20),(25,20),(5,5),(15,-7),(25,5),(15,20),(15,5),(5,-3),(40,-3),(5,35),(40,35),(5,20),(25,-3),(40,20),(25,35),(25,20)]#50,30
            #positions = [(5,-7),(25,-7),(5,20),(25,20),(5,5),(15,-7),(25,5),(15,20),(15,5)]#50
            #positions = [(5,-2),(32,-2),(5,27),(32,27),(5,12),(20,-2),(32,12),(20,27),(20,12)]#40
            #positions = [(5,-3),(40,-3),(5,35),(40,35),(5,20),(25,-3),(40,20),(25,35),(25,20)]#30
            #positions = [(15,-4)]
            for pos in positions:
                img = Image.new('L', (64, 64), color = 255)#create clean
                d = ImageDraw.Draw(img)
                d.text(pos, letter, font=font, fill= 0)#draw letter
                pix = np.array(img)
                npy_file.append(pix)
            
            fonts.append(font_file.split("_")[0].split(".")[0])#add font
            char.append(letter)#add character
            if "_B" in font_file: #add bold
                bold.append(1)
            else:
                bold.append(0)
            if "_I" in font_file: #add italics
                italics.append(1)
            else:
                italics.append(0)
        except:
            pass
                
npy_file = np.array(npy_file, dtype=np.float64)
npy_file = np.resize(npy_file, (len(npy_file), 64, 64, 1))
np.save("../data/clean_data.npy",npy_file)


pickle.dump( fonts, open( "../data/fonts.p", "wb" ) )
pickle.dump( char, open( "../data/char.p", "wb" ) )
pickle.dump( bold, open( "../data/bold.p", "wb" ) )
pickle.dump( italics, open( "../data/italics.p", "wb" ) )