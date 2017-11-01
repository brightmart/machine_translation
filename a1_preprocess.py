import codecs

def preprocess_english_file(file,file_save):
    #1.read file
    file_object = codecs.open(file, 'r', 'utf8')
    file_save_object = codecs.open(file_save, 'a', 'utf8')
    lines_cn = file_object.readlines()
    #2.replace
    for i,line in enumerate(lines_cn):
        if i%10000==0:
            print(i)
        line=line.lower().replace("\n"," ").replace("."," . ").replace(","," ,").replace("?"," ? ")
        file_save_object.write(line+"\n")
    #3.save

file='data/train.en'
file_save='data/train.en.processed'
preprocess_english_file(file,file_save)
