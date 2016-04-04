# -*- coding:utf-8 -*-

import re
import os
from shutil import copyfile
path=os.path.abspath('.')
newpath=path.replace('_posts','images')

filesInDirectory=[]
for root,dirs,files in os.walk('.',topdown=False):
    filesInDirectory=files
    
def markdown_syntax_verify(f_path,image_path_prefix):
    f=open(f_path,'r')
    replaced_dollar=re.sub(r'(?<!\$)(\$(?!\$)[^$]*(?<!\$)\$(?!\$))(?!\$)',r'\$\1$',f.read(),flags=re.MULTILINE)
    f.close()
    f=open(f_path,'w')
    replaced_image_path=re.sub(r'(!\[\w*\])\((\S*)\)',r'\1'+r'('+image_path_prefix+r'\2'+r')',replaced_dollar,flags=re.MULTILINE)
    f.write(replaced_image_path)
    f.close()

if not os.path.exists(newpath):
    os.makedirs(newpath)
    
image_path_prefix='/'+'/'.join(newpath.rsplit('/',3)[-3:])+'/'

for file in files:
    if file[-3:]=='png' or file[-3:]=='jpg':
        copyfile(path+'/'+file,newpath+'/'+file)
    if file[-2:]=='md':
        markdown_syntax_verify(file,image_path_prefix)
        

