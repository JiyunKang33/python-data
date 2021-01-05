#!/usr/bin/env python
# coding: utf-8

# # Ch1. 데이터에서 인사이트 발견하기

# ## 1. 탐색적 데이터 분석의 과정 
#  ###  1) 데이터의 출처와 주제에 대해 이해한다
#  ###  2) 데이터의 크기를 알아본다.
#  ###  3) 데이터의 구성 요소(피처)를 살펴본다
#  ###  4) 데이터의 속성 탐색하기. => 데이터에 질문을 던지기, 피처간의 상관관계 탐색, 수치형/범주형 피처 등
#  ###  5) 탐색한 데이터의 시각화 

# ## 2. 멕시코풍 프랜차이즈 chipotle의 주문 데이터 분석하기
# ### <Chipotle 데이터 셋의 기초정보 출력하기>

# In[21]:


import pandas as pd

file_path = '../Downloads/python-data-analysis-master/data/chipotle.tsv'

chipo = pd.read_csv(file_path, sep = '\t');  #csv파일이 탭으로 분리되어 있음

print(chipo.shape)  # shape : 데이터의 행과 열의 크기를 반환
print("-----------------")
print(chipo.info()) # info() : 행과 열의 구성정보를 나타냄


# ### <Chipotle 데이터 셋의 행과 열, 데이터 확인하기>

# In[29]:


chipo.head(10) # 데이터의 상위 10개 데이터를 보여줌 


# In[27]:


print(chipo.columns) # 컬럼정보


# In[28]:


print(chipo.index)


# #### ** order_id 는 숫자의 의미를 가지지 않기 때문에 str로 변경

# In[22]:


chipo['order_id'] = chipo['order_id'].astype(str)
print(chipo.info())


# In[31]:


chipo.describe()  # 수치형변수에 대해서 기초 통계량 확인


# In[32]:


chipo['quantity'].describe()  # 평균 주문 수량은 약 1.07 => 한 사람이 같은 메뉴를 여러개 구매하는 경우는 많지 않다!!


# #### ** unique 함수 : 범주형 변수에서 피처 내에 몇개의 범주가 있는지

# In[39]:


print(len(chipo['order_id'].unique()))
print(len(chipo['item_name'].unique()))


# #### ** value_counts() : 시리즈 객체에만 적용

# In[23]:


# 가장 많이 주문한 아이템 top10 출력
item_count = chipo['item_name'].value_counts()[:10]
print(item_count)
print("-----------------------")
for idx, (val, cnt) in enumerate(item_count.iteritems(),1) :
    print("Top", idx, ":", val, cnt)


# #### ** groupby() : 특정 피처 기준 그룹별 연산

# In[24]:


# 아이템별 주문개수 구하기
order_count = chipo.groupby('item_name')['order_id'].count()
order_count[:10]


# In[25]:


# 아이템별 주문총량 구하기
item_quantity = chipo.groupby('item_name')['quantity'].sum()
item_quantity[:10]


# ### <시각화>

# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

# 아이템별 주문 총량 시각화
item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cnt = item_quantity.values.tolist()

plt.bar(x_pos, order_cnt, align = 'center')
plt.ylabel('order_item_count')
plt.title('Distribution of all ordered item')

plt.show()


# ### <데이터 전처리>

# In[73]:


print(chipo.info())
print("------------------")
chipo['item_price'].head()


# In[26]:


# item_price 컬럼에서 $를 제거하고 수치 데이터로 변경
chipo['item_price'] = chipo['item_price'].apply(lambda x : float(x[1:]))
chipo.describe()


# ### <탐색적 분석 : 스무고개로 개념적 탐색 분석하기>

# In[27]:


# 주문당 평균 계산금액 출력하기
chipo.groupby('order_id')['item_price'].sum().mean()


# In[28]:


# 한 주문에 10달러 이상 지불한 주문번호 출력하기
chipo_orderid_group = chipo.groupby('order_id').sum() # 주문당 총 지불금액
chipo_orderid_group


# In[29]:


result = chipo_orderid_group[chipo_orderid_group['item_price'] >= 10]
print(result[:10])
print(result.index.values)


# In[53]:


# 각 아이템의 가격 구하기
chipo_oneitem = chipo[chipo.quantity ==1] #주문수량이 1개인 경우
price_peritem = chipo_oneitem.groupby('item_name').min() # 각 그룹별 최저가 계산
price_peritem

# 정렬. sort_values(by = '정렬기준 값', ascending = '오름차순(True)/내리차순(False)')
price_peritem.sort_values(by = 'item_price', ascending = False)[:10]


# In[62]:


# 시각화
price_peritem
item_name_list = price_peritem.index.tolist()
x_pos = np.arange(len(item_name_list))
item_price = price_peritem['item_price'].tolist()


plt.bar(x_pos, item_price, align='center')
plt.ylabel('item_price($)')
plt.title('Distribution of item price')
plt.show()


# In[63]:


plt.hist(item_price)
plt.ylabel('counts')
plt.title('Histogram of item price')
plt.show()


# In[66]:


# 가장 비싼 주문에서 아이템이 총 몇개 팔렸는지

chipo.groupby('order_id').sum().sort_values(by = 'item_price', ascending = False)[:5]


# #### ** drop_duplicates  : 중복제거

# In[72]:


# Veggie Salad Bowl 이 몇번 주문되었는지
chipo_salad = chipo[chipo['item_name'] == 'Veggie Salad Bowl']
chipo_salad 
print(len(chipo_salad))

# 한 주문내에 중복 집계된 item_name을 제거
chipo_salad = chipo_salad.drop_duplicates(['item_name','order_id'])
chipo_salad
print(len(chipo_salad))
chipo_salad.head(5)


# In[77]:


# Chicken Bowl을 두개 이상 주문한 주문 횟수 구하기
chipo_chicken = chipo[chipo['item_name'] == 'Chicken Bowl']
chipo_chicken

chipo_chicken_ordersum = chipo_chicken.groupby('order_id').sum()['quantity']
chipo_chicken_ordersum

chipo_chicken_result = chipo_chicken_ordersum[chipo_chicken_ordersum >=2]
chipo_chicken_result

print(len(chipo_chicken_result))
chipo_chicken_result.head(5)


# In[ ]:




