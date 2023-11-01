from Predict import Predict_disease

img,classes = Predict_disease(r'C:/Users/nattanan.t/Desktop/Personal_Files/FMS_ID/rice-blast-disease.jpg')

img.save('result.jpg')
print(classes)  
    
