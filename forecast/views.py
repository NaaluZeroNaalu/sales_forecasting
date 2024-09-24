import pandas as pd
from django.shortcuts import render
from .forms import CSVUploadForm
from .models import CSVFile
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def Index(request):
    form = CSVUploadForm(request.POST, request.FILES)
    plot_data = None
    error_message = None
    highest_sales_month = None
    highest_sales_value = None
    lowest_sales_month = None
    lowest_sales_value = None

    if request.method == 'POST':
        if form.is_valid():
            form.save()
            latest_file = CSVFile.objects.latest('uploaded_at')
            csv_file_path = latest_file.file.path

            try:
             
                df = pd.read_csv(csv_file_path)

                df['MONTH'] = df['MONTH'].str.upper().str.strip()

              
                month_mapping = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                df['month_num'] = df['MONTH'].map(month_mapping)

                
                historical_sales = df.groupby('month_num')['SALES'].sum().reindex(range(1, 13), fill_value=0)

               
                highest_sales_value = historical_sales.max()
                lowest_sales_value = historical_sales.min()
                highest_sales_month = historical_sales.idxmax()
                lowest_sales_month = historical_sales.idxmin()

          
                X = df[['month_num']]
                y = df['SALES']

             
                if y.isnull().any():
                    imputer = SimpleImputer(strategy='mean')
                    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

          
                model = LinearRegression()
                model.fit(X, y)

        
                future_months = pd.DataFrame({'month_num': range(1, 13)})
                future_sales = model.predict(future_months)

              
                plot_data = {
                    'months': list(month_mapping.keys()),
                    'historical_sales': historical_sales.tolist(),
                    'forecasted_sales': future_sales.tolist()
                }

            except Exception as e:
                error_message = str(e)

    return render(request, 'index.html', {
        'form': form,
        'plot_data': plot_data,
        'error_message': error_message,
        'highest_sales_month': list(month_mapping.keys())[highest_sales_month - 1] if highest_sales_month else None,
        'highest_sales_value': highest_sales_value if highest_sales_value is not None else 0,
        'lowest_sales_month': list(month_mapping.keys())[lowest_sales_month - 1] if lowest_sales_month else None,
        'lowest_sales_value': lowest_sales_value if lowest_sales_value is not None else 0,
    })
