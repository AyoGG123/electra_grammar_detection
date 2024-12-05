import torch
import torch.nn.functional as F
import re

temp = '以色列國防軍已經結束動員，從18萬人急遽擴大到54萬人，不僅圍住加沙'
temp = re.sub(r' ', '', temp)
print(temp)
temp = re.split("，|。|？|！|；", temp)
print(temp)
