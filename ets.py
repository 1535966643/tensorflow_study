
from aip import AipOcr
""" 你的 APPID AK SK """
APP_ID = '15180273'
API_KEY = 'ztYBSs4r45oVMsmn5ZAWESfE'
SECRET_KEY = 'dtUXqOgGGSDNShPTD8YyDR0Rz55fBb0w'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
image = get_file_content('c:/car.jpg')
""" 如果有可选参数 """
options = {}
options["language_type"] = "CHN_ENG"
options["detect_direction"] = "true"
options["detect_language"] = "true"
options["probability"] = "true"
# """ 带参数调用通用文字识别, 图片参数为本地图片 """
text = client.basicGeneral(image)
# print(text['words_result'])
for x in text['words_result']:
	print(x['words'])
# print(text['words_result'])

