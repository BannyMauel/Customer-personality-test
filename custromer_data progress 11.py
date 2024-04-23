import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#DATA EXPLORATION


def main():
    df=pd.read_csv('Customer Personality Analysis.csv')
    pd.options.display.max_rows=2240
    #print(df.head())

#fill NaN values to 0
    df.fillna(0,inplace=True)

#AVERAGE INCOME
    df_mean=df['Income'].mean()
#print(df_mean)

#replace null values with average of income
    for x in df.index:
        if df.loc[x,'Income']==0:
            df.loc[x,'Income']=df_mean
    #print(df.head())
#CUSTOMER DEMOGRAPHICS
    #CATEGORIZATION

    #Distribution by education level, marital status, birth year
    #average income of marital status
    #     #SINGLE
    single_df= df[df['Marital_Status'] == 'Single'][['Marital_Status', 'Income']]
    single_avg_income = df[df['Marital_Status'] == 'Single']['Income'].mean()
    # #print(single_df.head())
    # print(single_avg_income)

    #     #MARRIED
    married_df=df[df['Marital_Status']=='Married'][['Marital_Status', 'Income']]
    married_avg_income=df[df['Marital_Status'] == 'Married']['Income'].mean()
    # print(married_avg_income)
    # #print(married_df.head())

    #     #TOGETHER
    Together_df=df[df['Marital_Status']=='Together'][['Marital_Status','Income']]
    Together_avg_income=df[df['Marital_Status']=='Together']['Income'].mean()
    # print(Together_avg_income)

    #     #DIVORCED
    Divorced_df=df[df['Marital_Status']=='Divorced'][['Marital_Status','Income']]
    Divorce_avg_income=df[df['Marital_Status']=='Divorced']['Income'].mean()
    # print(Divorce_avg_income)
    # #print(Divorced_df.head())

    #     #WIDOW
    Widow_df=df[df['Marital_Status']=='Widow'][['Marital_Status','Income']]
    Widow_avg_income=df[df['Marital_Status']=='Widow']['Income'].mean()
    # print(Widow_avg_income)
    # #print(Widow_df.head())
    
    #     #ALONE
    alone_df=df[df['Marital_Status']=='Alone'][['Marital_Status', 'Income']] 
    alone_avg_income=df[df['Marital_Status']=='Alone']['Income'].mean()
    # print(alone_avg_income)
    # #print(alone_df.head())   
    
    #     #ABSURD
    absurd_df=df[df['Marital_Status']=='Absurd'][['Marital_Status','Income']]
    absurd_avg_income=df[df['Marital_Status']=='Absurd']['Income'].mean()
    #print(absurd_df)
    #print(absurd_avg_income)

    # #YOLO
    YOLO_df=df[df['Marital_Status']=='YOLO'][['Marital_Status','Income']]
    YOLO_avg_income=df[df['Marital_Status']=='YOLO']['Income'].mean()
    # #print(YOLO_df)
    # print(YOLO_avg_income)

    #AVERAGE INCOME FOR MARITAL STATUS DISTRIBUTION
    # Calculate average income for each marital status
    avg_income_by_marital_status = df.groupby('Marital_Status')['Income'].mean().reset_index()

    # Merge the average income with the original DataFrame
    df = pd.merge(df, avg_income_by_marital_status, on='Marital_Status', suffixes=('', '_avg'))

    # Rename the columns
    df.rename(columns={'Income_avg': 'Avg_Income'}, inplace=True)
    #print(df.head())

    #AVERAGE INCOME FOR EDUCATION DISTRIBUTION
    avg_income_by_education = df.groupby('Education')['Income'].mean().reset_index()

    # Merge the average income with the original DataFrame
    df = pd.merge(df, avg_income_by_education, on='Education', suffixes=('', '_avg'))

    # Rename the columns
    df.rename(columns={'Income_avg': 'Avg_Income'}, inplace=True)
    #print(df.head())
    
    #Distribution of customers by their education level, marital status, year birth
    #MARITAL STATUS
    plt.bar(avg_income_by_marital_status['Marital_Status'], avg_income_by_marital_status['Income'], color='skyblue')
    plt.title('Average Income by Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Average Income')

    # #EDUCATION
    plt.bar(avg_income_by_education['Education'], avg_income_by_education['Income'], color='skyblue')
    plt.title('Average Income by Marital Status')
    plt.xlabel('Education')
    plt.ylabel('Average Income')

    #YEAR BIRTH
    plt.hist(df['Year_Birth'])
    plt.xlabel('Year birth')
    plt.title('Marital status Distribution')
    
    #plt.show()

    #AVERAGE INCOME OF CUSTOMERS:
    avg_income=np.mean(df['Income'])
    #print(avg_income)

#CUSTOMER BEHAVIOR
    # childen at home
   
    df.loc[df['Teenhome']==0, 'Teen_home']='no child'
    df.loc[df['Teenhome']!=0, 'Teen_home']= 'child'

    # kid at home
    df.loc[df['Kidhome']==0, 'kid_home']='no child'
    df.loc[df['Kidhome']!=0, 'kid_home']= 'child'
    
    
    #average income spent on wine,meat,fruit,fish,sweetproducts,goldprod
    avg_income_wine=np.mean(df['MntWines'])
    #print(avg_income_win)

    avg_income_meat=np.mean(df['MntMeatProducts'])
    #print(avg_income_meat)

    avg_income_fruit=np.mean(df['MntFruits'])
    #print(avg_income_fruit)

    avg_income_fish=np.mean(df['MntFishProducts'])
    #print(avg_income_fish)

    avg_income_sweet=np.mean(df['MntSweetProducts'])
    #print(avg_income_sweet)

    avg_income_gold=np.mean(df['MntGoldProds'])
    #print(avg_income_gold)

    #Accepted campaigns
    #Acceptedcmp1
    df.loc[df['AcceptedCmp1']==0, 'Num_acceptedCmp1']='not accepted'
    df.loc[df['AcceptedCmp1']!=0, 'Num_acceptedCmp1']=' accepted'
    

    #Acceptedcmp2
    df.loc[df['AcceptedCmp2']==0, 'Num_acceptedCmp2']='not accepted'
    df.loc[df['AcceptedCmp2']!=0, 'Num_acceptedCmp2']=' accepted'

    #Acceptedcmp3
    df.loc[df['AcceptedCmp3']==0, 'Num_acceptedCmp3']='not accepted'
    df.loc[df['AcceptedCmp3']!=0, 'Num_acceptedCmp3']=' accepted'
    

    #Acceptedcmp4
    df.loc[df['AcceptedCmp4']==0, 'Num_acceptedCmp4']='not accepted'
    df.loc[df['AcceptedCmp4']!=0, 'Num_acceptedCmp4']=' accepted'

    #Acceptedcmp5
    df.loc[df['AcceptedCmp5']==0, 'Num_acceptedCmp5']='not accepted'
    df.loc[df['AcceptedCmp5']!=0, 'Num_acceptedCmp5']=' accepted'

    

    #customer complains
    df.loc[df['Complain']==0, 'Num_complains']=' no complains'
    df.loc[df['Complain']!=0, 'Num_complains']=' complains'
    
    pd.options.display.max_columns=37
    
    #print(df.head())

#PURCHASE PATTERNS

    #number of times customer visit website
    mean_webvisit=np.mean(df['NumWebVisitsMonth'])
    #print(mean_webvisit)

    #Website visit per month by education level
    avg_website_visit=df.groupby('Education')['NumWebVisitsMonth'].sum().reset_index()
    #print(avg_website_visit)

    df = pd.merge(df, avg_website_visit, on='Education', suffixes=('', '_avg'))
    df.rename(columns={'NumWebVisitsMonth_avg': 'Avg_NumWebVisitsMonth'}, inplace=True)
    #print(df.head())

    plt.bar(avg_website_visit['Education'], avg_website_visit['NumWebVisitsMonth'], color='skyblue')
    plt.title('Average website visit by education')
    plt.xlabel('Education')
    plt.ylabel('Average website visit')
    #plt.show()

    corre_purchase_ppt1=np.corrcoef(df['NumDealsPurchases'],df['NumWebPurchases'])[0,1]
    corre_purchase_ppt2=np.corrcoef(df['NumCatalogPurchases'],df['NumStorePurchases'])[0,1]

    # print(corre_purchase_ppt1)
    # print(corre_purchase_ppt2)
    
#TIME ANALYSIS: 1st Approach
    from datetime import datetime

# Assuming the database was created on January 1st, 2012
    database_creation_date = datetime(2012, 1, 1)

# Current date
    current_date = datetime.now()

# Calculate the duration
    duration = current_date - database_creation_date

    #print("The database has been stored for:", duration)


#TIME ANALYSIS: 2nd approach
#How long have customers been in the database (Dt_Customer)
    
    # Convert 'Date' column to datetime format
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
  

    # # Extract year from 'Date' column
    df['Year'] = df['Dt_Customer'].dt.year
    # #print(df.head(5))

    #Drop the original 'Date' column if not needed
    df.drop(columns=['Dt_Customer'], inplace=True)

    
    df.loc[df['Year']==2012, 'Dt_Customer_stay']='12 years'
    df.loc[df['Year']!=2012, 'Dt_Customer_stay']='less than 12 years'
    #print(df.head(6))    



#correlation and relationship

#correlation btw receny and response   
    corre_ren=np.corrcoef(df['Recency'],df['Response'])[0,1]
    #print(corre_ren)


    #customer xtrics
    corre=np.corrcoef(df['Income'],df['MntFruits'])[0,1]
    corre1=np.corrcoef(df['Income'],df['MntWines'])[0,1]
    
    #print(corre)
    #print(corre1)


    #response & campaigns
    corre_rps_cmp1=np.corrcoef(df['Response'],df['AcceptedCmp1'])[0,1]
    corre_rps_cmp2=np.corrcoef(df['Response'],df['AcceptedCmp2'])[0,1]
    corre_rps_cmp3=np.corrcoef(df['Response'],df['AcceptedCmp3'])[0,1]
    corre_rps_cmp4=np.corrcoef(df['Response'],df['AcceptedCmp4'])[0,1]
    corre_rps_cmp5=np.corrcoef(df['Response'],df['AcceptedCmp5'])[0,1]
    
    #print(corre_rps_cmp1)
    #print(corre_rps_cmp2)
    #print(corre_rps_cmp3)
    #print(corre_rps_cmp4)
    #print(corre_rps_cmp5)

#CALCULATION OF PROFIT
    total_cost=np.sum(df['Z_CostContact'])
    #print(total_cost)

    total_revenue=np.sum(df['Z_Revenue'])
    #print(total_revenue)

    profit=total_revenue-total_cost
    #print(profit)
    

    



main()