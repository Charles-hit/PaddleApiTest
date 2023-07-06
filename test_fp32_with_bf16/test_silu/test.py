import os
import sys

# print ('***获取当前目录***')
# print (os.getcwd())
# print (os.path.abspath(os.path.dirname(__file__)))


# print ('***获取上级目录***')
# print (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(...)
# print (os.path.abspath(os.path.dirname(os.getcwd())))
# os.path.abspath(os.path.dirname(os.getcwd()))
# sys.path.append(...)
# print (os.path.abspath(os.path.join(os.getcwd(), "..")))
# os.path.abspath(os.path.join(os.getcwd(), ".."))
# sys.path.append(print (os.path.abspath(os.path.join(os.getcwd(), ".."))))

print ('***获取上上级目录***')
print (os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))