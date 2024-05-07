from flask import Flask, render_template, request
from flask import jsonify
from flask.json import dumps
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px  

# chatbot_module.py
import pandas as pd
from nltk.tokenize import word_tokenize

def tokenize_and_filter(text):
    tokens = word_tokenize(text)
#     tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    print(tokens)
    return tokens


# Execute query function
def execute_query(query, merged_data):
    tokens = tokenize_and_filter(query)
#     result = []
    brand_names = ['Samsung', 'Infinix', 'Redmi', 'Tecno', 'Honor', "Vivo", 'OPPO', 'Itel', 'Realme']
    
    
    
    # Combination of price and rating without brand
    if 'price' in tokens and 'rating' in tokens and not any(brand_name.lower() in tokens for brand_name in brand_names):
        print('Combination of price and rating without brand')
        min_price, max_price, min_rating, max_rating = None, None, None, None
        if 'under' in tokens:
            max_price = float(tokens[tokens.index('under') + 1])
        elif 'above' in tokens:
            min_price = float(tokens[tokens.index('above') + 1])
        elif 'between' in tokens:
            min_price, max_price = float(tokens[tokens.index('between') + 1]), float(tokens[tokens.index('and') + 1])

        if 'and' in tokens:
            index_and = tokens.index('and')
            if 'under' in tokens[index_and:]:
                print('yes')
                max_rating = float(tokens[index_and + tokens[index_and:].index('under') + 1])
            elif 'above' in tokens[index_and:]:
                min_rating = float(tokens[index_and + tokens[index_and:].index('above') + 1])
            elif 'between' in tokens[index_and:]:
                min_rating, max_rating = float(tokens[index_and + tokens[index_and:].index('between') + 1]), float(tokens[index_and + tokens[index_and:].index('and') + 1])

        print(index_and)
        print(max_rating, min_rating)
        result = merged_data[
            ((min_price is None) or (merged_data['Price'] > min_price)) &
            ((max_price is None) or (merged_data['Price'] < max_price)) &
            ((min_rating is None) or (merged_data['Rating'] > min_rating)) &
            ((max_rating is None) or (merged_data['Rating'] < max_rating))
        ]
        

# Combination of price and rating with brand
    elif 'price' in tokens and 'rating' in tokens and any(brand_name.lower() in tokens for brand_name in brand_names):
        brand = ''
        for brand_name in brand_names:
            if brand_name.lower() in tokens:
                brand = brand_name
                break

        min_price, max_price, min_rating, max_rating = None, None, None, None
        if 'under' in tokens:
            max_price = float(tokens[tokens.index('under') + 1])
        elif 'above' in tokens:
            min_price = float(tokens[tokens.index('above') + 1])
        elif 'between' in tokens:
            min_price, max_price = float(tokens[tokens.index('between') + 1]), float(tokens[tokens.index('and') + 1])

        if 'and' in tokens:
            index_and = tokens.index('and')
            if 'under' in tokens[index_and:]:
                max_rating = float(tokens[index_and + tokens[index_and:].index('under') + 1])
            elif 'above' in tokens[index_and:]:
                min_rating = float(tokens[index_and + tokens[index_and:].index('above') + 1])
            elif 'between' in tokens[index_and:]:
                min_rating, max_rating = float(tokens[index_and + tokens[index_and:].index('between') + 1]), float(tokens[index_and + tokens[index_and:].index('and') + 1])

        result = merged_data[
            ((min_price is None) or (merged_data['Price'] >= min_price)) &
            ((max_price is None) or (merged_data['Price'] <= max_price)) &
            ((min_rating is None) or (merged_data['Rating'] >= min_rating)) &
            ((max_rating is None) or (merged_data['Rating'] <= max_rating)) &
            (merged_data['Company'] == brand)
        ]

            
    
    
#only catering price without brands
    elif 'price' in tokens and not any(brand_name.lower() in tokens for brand_name in brand_names):
        min_price, max_price = None, None
        if 'under' in tokens:
            if 'k' in tokens[tokens.index('under') + 1]:
                str = tokens[tokens.index('under') + 1]
                max_price = float(str.replace("k","000"))
            else:
                max_price = float(tokens[tokens.index('under') + 1])
            result = merged_data[(merged_data['Price'] <= max_price)]
        elif 'above' in tokens:
            if 'k' in tokens[tokens.index('above') + 1]:
                str = tokens[tokens.index('above') + 1]
                min_price = float(str.replace("k","000"))
            else:
                min_price = float(tokens[tokens.index('above') + 1])
            result = merged_data[(merged_data['Price'] >= min_price)]
        else:
            if 'k' in tokens[tokens.index('between') + 1] and 'k' in tokens[tokens.index('and') + 1]:
                str1 = tokens[tokens.index('between') + 1]
                min_price = float(str1.replace("k","000"))
                str2 = tokens[tokens.index('and') + 1]
                max_price = float(str2.replace("k","000"))
            else:
                min_price, max_price = float(tokens[tokens.index('between') + 1]), float(tokens[tokens.index('and') + 1])
            result = merged_data[(merged_data['Price'] >= min_price) & (merged_data['Price'] <= max_price)]
            
            

#catering price with brands
    elif 'price' in tokens and any(brand_name.lower() in tokens for brand_name in brand_names):
        brand = ''
        for brand_name in brand_names:
            if brand_name.lower() in tokens:
                brand = brand_name
                break
                
        min_price, max_price = None, None
        if 'under' in tokens:
            if 'k' in tokens[tokens.index('under') + 1]:
                str = tokens[tokens.index('under') + 1]
                max_price = float(str.replace("k","000"))
            else:
                max_price = float(tokens[tokens.index('under') + 1])
            result = merged_data[(merged_data['Price'] <= max_price) & (merged_data['Company'] == brand)]
        elif 'above' in tokens:
            if 'k' in tokens[tokens.index('above') + 1]:
                str = tokens[tokens.index('above') + 1]
                min_price = float(str.replace("k","000"))
            else:
                min_price = float(tokens[tokens.index('above') + 1])
            result = merged_data[(merged_data['Price'] >= min_price) & (merged_data['Company'] == brand)]
        else:
            if 'k' in tokens[tokens.index('between') + 1] and 'k' in tokens[tokens.index('and') + 1]:
                str1 = tokens[tokens.index('between') + 1]
                min_price = float(str1.replace("k","000"))
                str2 = tokens[tokens.index('and') + 1]
                max_price = float(str2.replace("k","000"))
            else:
                min_price, max_price = float(tokens[tokens.index('between') + 1]), float(tokens[tokens.index('and') + 1])
            result = merged_data[(merged_data['Price'] >= min_price) & (merged_data['Price'] <= max_price) & (merged_data['Company'] == brand)]

            

#only catering rating without brands     
    elif 'rating' in tokens and not any(brand_name.lower() in tokens for brand_name in brand_names):
        min_rating, max_rating = None, None
        if 'under' in tokens:
            max_rating = float(tokens[tokens.index('under') + 1])
            result = merged_data[merged_data['Rating'] < max_rating]
        elif 'above' in tokens:
            min_rating = float(tokens[tokens.index('above') + 1])
            result = merged_data[merged_data['Rating'] > min_rating]
        elif 'between' in tokens:
            min_rating, max_rating = float(tokens[tokens.index('between') + 1]), float(tokens[tokens.index('and') + 1])
            result = merged_data[(merged_data['Rating'] > min_rating) & (merged_data['Rating'] < max_rating)]
        else:
            rating = float(tokens[tokens.index('rating') + 1])
            result = merged_data[merged_data['Rating'] == rating]
            
            

#catering rating with brands
    elif 'rating' in tokens and any(brand_name.lower() in tokens for brand_name in brand_names):
        for brand_name in brand_names:
            if brand_name.lower() in tokens:
                brand = brand_name
                break
        
        min_rating, max_rating = None, None
        if 'under' in tokens:
            max_rating = float(tokens[tokens.index('under') + 1])
            result = merged_data[(merged_data['Rating'] < max_rating) & (merged_data['Company'] == brand)]
        elif 'above' in tokens:
            min_rating = float(tokens[tokens.index('above') + 1])
            result = merged_data[(merged_data['Rating'] > min_rating) & (merged_data['Company'] == brand)]
        elif 'between' in tokens:
            min_rating, max_rating = float(tokens[tokens.index('between') + 1]), float(tokens[tokens.index('and') + 1])
            result = merged_data[(merged_data['Rating'] > min_rating) & (merged_data['Rating'] < max_rating) & (merged_data['Company'] == brand)]
        else:
            rating = float(tokens[tokens.index('rating') + 1])
            result = merged_data[(merged_data['Rating'] == rating) & (merged_data['Company'] == brand)]

            
            
#Best or Top Product without catering brand
    elif ('top' in tokens or 'best' in tokens) and not any(brand_name.lower() in tokens for brand_name in brand_names):
        temp = merged_data[(merged_data['Rating'] == 5)]
        result = temp.head(5)
        
#Best or Top Product with brand    
    elif ('top' in tokens or 'best' in tokens) and any(brand_name.lower() in tokens for brand_name in brand_names):
        for brand_name in brand_names:
            if brand_name.lower() in tokens:
                brand = brand_name
                break
                
        filtered_data = merged_data[merged_data['Company'] == brand]
        filtered_data['Review Length'] = filtered_data['Review Content'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        sorted_data = filtered_data.sort_values(by=['Rating', 'Review Length'], ascending=[False, False])
        result = sorted_data.head(5)
        
    
    
    else:
        temp = []
        for _, product in merged_data.iterrows():
            count_matching_words = sum(token in product['Name'].lower() for token in tokens)
            if count_matching_words > 0:
                product_with_count = product.copy()
                product_with_count['CountMatchingWords'] = count_matching_words

                temp.append(product_with_count)

        matched = pd.DataFrame(temp)
    
        max_count = matched['CountMatchingWords'].max()
        result = matched[matched['CountMatchingWords'] == max_count]



    response = "Filtered products based on your query:\n"

    for _, product in result.iterrows():
        # response += f"\n Product: {product['Name']}\n Brand: {product['Company']}\n Price: {product['Price']}\n Rating: {product['Rating']}\n Link: {product['Link']}\n"
        response += f"\n Product: {product['Name']}\n Brand: {product['Company']}\n Price: {product['Price']}\n Rating: {product['Rating']}\n"

    return response




app = Flask(__name__)

# Load your CSV files
products_df = pd.read_csv("product_info.csv", index_col="ID")
reviews_df = pd.read_csv("rev_info.csv", index_col="ID")
reviews_df = reviews_df.drop(columns="Name")

merged_data = pd.merge(products_df, reviews_df, on='ID')  


merged_data['Price'] = merged_data['Price'].str.replace('Rs. ', '').str.replace(',', '').astype(float)

merged_data['Rating'] = merged_data['Rating'].replace('-', float('nan'))
merged_data['Rating'] = pd.to_numeric(merged_data['Rating'], errors='coerce')


@app.route('/')
def index():
    return render_template('index.html', merged_data=merged_data)


@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    result = {}  # Initialize result to an empty dictionary

    # Basic statistics (calculated regardless of the request method)
    result['total_listings'] = len(merged_data)
    result['avg_price'] = merged_data['Price'].mean()
    result['avg_rating'] = merged_data['Rating'].mean()
    result['avg_review_count'] = 12
    result['total_questions'] = 0  
    result['No_of_brands'] = 11

    # Top 5 products based on average rating (calculated regardless of the request method)
    # top_products = merged_data.groupby('Name').agg({'Rating': 'mean', 'Review ID': 'count'}).reset_index()
    # top_products = top_products.sort_values(by=['Rating', 'Review ID'], ascending=[False, False]).head(5)

    # Top 5 products based on average rating (calculated regardless of the request method)
    top_products = merged_data.groupby(['Name', 'Link']).agg({'Rating': 'mean', 'Review ID': 'count'}).reset_index()
    top_products = top_products.sort_values(by=['Rating', 'Review ID'], ascending=[False, False]).head(5)

    result['top_products'] = top_products

    # Create a bar chart for top products (calculated regardless of the request method)
    fig = px.bar(top_products, x='Name', y='Rating', color='Review ID', title='Top 5 Products')
    result['chart_div'] = fig.to_html(full_html=False)  # Pass the Plotly chart div to the template

    # Extract data for the "Company vs. Price" graph from the merged_data
    graph_data = merged_data[['Company', 'Price']].to_json(orient='records')
    result['graph_data'] = graph_data

    # Extract data for the "Company vs. Rating" graph from the merged_data
    rating_graph_data = merged_data[['Company', 'Rating']].to_json(orient='records')
    result['rating_graph_data'] = rating_graph_data

    if request.method == 'POST':
        query = request.form['query']
        chatbot_response = execute_query(query, merged_data).replace('\n', '<br>')
        result['query'] = query
        result['chatbot_response'] = chatbot_response
    return render_template('dashboard.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

    