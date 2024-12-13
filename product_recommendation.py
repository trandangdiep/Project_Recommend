import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, CoClustering, BaselineOnly
from surprise.model_selection.validation import cross_validate

# Function to get recommendations based on cosine similarity
def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    # Ensure product IDs match the indices of cosine similarity matrix
    # Find the index of the selected product in df_products
    idx = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    
    if not idx:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame
    
    idx = idx[0]  # Get the first index if there are multiple matches
    
    # Ensure that the index is within the bounds of the cosine similarity matrix
    if idx >= len(cosine_sim):
        print(f"Index {idx} is out of bounds for cosine similarity matrix.")
        return pd.DataFrame()  # Return an empty DataFrame if index is out of bounds
    
    # Get cosine similarity scores for the selected product
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort products based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the first product as it's the same one we selected
    sim_scores = sim_scores[1:nums+1]
    
    # Extract the indices of the most similar products
    product_indices = [i[0] for i in sim_scores]
    
    # Return the recommended products
    return df.iloc[product_indices]
# hàm gợi ý sản phẩm và lọc theo điều kiện
def display_recommended_products_1(recommended_products,ma_san_pham,df_output,cols=5):
    recommended_products = recommended_products[recommended_products["diem_trung_binh"] >= 4.5]
    recommended_products = recommended_products[recommended_products['ma_san_pham'] != ma_san_pham]
    if len(recommended_products) > 3:
        recommended_products = recommended_products[0:3]
    for i in range(0, len(recommended_products), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    with st.container(border=True):
                        st.image(product['hinh_anh'])
                        st.write(product['ten_san_pham'])
                        st.write("Mã sản phẩm :",product['ma_san_pham'])
                        st.write("Giá :",product['gia_ban'],' VNĐ')
                        st.write("Sao :",product['diem_trung_binh'],' ⭐')
                        luotmua = df_output[df_output['ma_san_pham']==product['ma_san_pham']].groupby('ma_san_pham',as_index=False)['ma_khach_hang'].count()
                        if len(luotmua) > 0:
                            st.write("Lượt mua: ", luotmua['ma_khach_hang'][0]," 🛒")
                        else:
                            st.write("Lượt mua: 0 🛒")
                        expander=st.expander("Mô tả")
                        truncated_description = ' '.join(product['mo_ta'].split()[:100]) + '...'
                        expander.write(truncated_description)
                        # Nút Xem thêm
                        xem_them_key = f"xem_them_{product['ma_san_pham']}"
                        if st.button("Xem thêm", key=xem_them_key):
                            # Cập nhật sản phẩm được chọn
                            st.session_state.selected_ma_san_pham = product['ma_san_pham']
                            st.session_state['selected_product'] = product
                            # Làm mới trang
                            st.rerun()



# hàm gợi ý sản phẩm tượng tự 
def display_recommended_products(recommended_products,df_output,cols=5):
    for i in range(0, len(recommended_products), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    with st.container(border=True):
                        st.image(product['hinh_anh'])
                        st.write(product['ten_san_pham'])
                        st.write("Mã sản phẩm :",product['ma_san_pham'])
                        st.write("Giá :",product['gia_ban'],' VNĐ')
                        st.write("sao :",product['diem_trung_binh'],' ⭐')
                        luotmua = df_output[df_output['ma_san_pham']==product['ma_san_pham']].groupby('ma_san_pham',as_index=False)['ma_khach_hang'].count()
                        if len(luotmua) > 0:
                            st.write("Lượt mua: ", luotmua['ma_khach_hang'][0]," 🛒")
                        else:
                            st.write("Lượt mua: 0 🛒")
                        expander=st.expander("Mô tả")
                        truncated_description = ' '.join(product['mo_ta'].split()[:100]) + '...'
                        expander.write(truncated_description)
# recommended collab
def recommended_collab(userId,df_sub,df_select,algorithm_KNN):
    # #loại bỏ những sản phẩm đã được mua
    df_list= df_select['ma_san_pham'].tolist()
    df_score = df_sub[~df_sub['ma_san_pham'].isin(df_list)]
    df_score.reset_index(drop=True,inplace=True)

    #tạo cột và dự đoán các giá trị
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: algorithm_KNN.predict(userId, x).est) # est: get EstimateScore

    #sắp xếp lại các giá trị theo thứ tự giảm dần
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)

    #xóa dữ liệu trùng
    df_score = df_score.drop_duplicates(subset='ma_san_pham', keep='first')
    #lọc những dòng có EstimateScore >= 4
    df_score = df_score[df_score['EstimateScore'] >= 4].head(9)
    return df_score
    


# hàm gợi ý sản phẩm theo người dùng
def Login_show(recommended_collab,display_recommended_products,df_customer,df_output,df_products,selected_username,algorithm_KNN):
    # lấy id
    get_id = df_customer[df_customer['username']==selected_username[0]]['ma_khach_hang'].reset_index(drop='index')[0]
    # từ id đã có , lấy sản phẩm
    customer_products = df_output[df_output['ma_khach_hang'] == get_id]['ma_san_pham'].tolist()
    if customer_products:
        st.write("### Sản phẩm đã mua:")
        # từ id sản phẩm , lấy thông tin sản phẩm
        purchased_products = df_products[df_products['ma_san_pham'].isin(customer_products)]
        st.dataframe(purchased_products[['ma_san_pham', 'ten_san_pham']])

        st.write("### Gợi ý sản phẩm liên quan:")
        recommendations = recommended_collab(
        get_id,
        df_output,
        purchased_products,
        algorithm_KNN
        )
        # st.dataframe(recommendations)
        df_list = recommendations['ma_san_pham'].tolist()
        recommendations =  df_products[df_products['ma_san_pham'].isin(df_list)]
        display_recommended_products(recommendations,df_output, cols=3)
    else:
        st.info(f"Khách hàng **{selected_username[2]}** chưa mua sản phẩm nào.")
        st.write("### Gợi ý sản phẩm đang hot:")

        # lấy ra 9 sản phẩm được mua nhiều nhất
        df_top9 = df_output.groupby('ma_san_pham',as_index=False)['so_sao'].count().sort_values(by='so_sao').tail(9)

        # lấy điểm trung bình của top 9 sản phẩm
        top9_merged = pd.merge(df_top9, df_products, on="ma_san_pham",how="left")
        #lọc lấy những sản phẩm có điểm trung bình >= 4.5
        top9_merged = top9_merged[top9_merged['diem_trung_binh'] >= 4.5]
        display_recommended_products(top9_merged,df_output, cols=3)


        
    
# Read data
df_products = pd.read_csv('data/Product.csv')
df_customer = pd.read_csv('data/Customer1.csv')
df_output = pd.read_csv('data/Danh_gia1.csv')

# Load precomputed cosine similarity
with open('model/products_cosine_sim_.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

# Load model surprise Knn
with open('model/surprise_knn_model.pkl', 'rb') as f:
    algorithm_KNN = pickle.load(f)

###### Streamlit Interface ######

    # GUI
st.title("Data Science Project")
st.write("## Product Recommendation")

menu = ["Business Objective", "Build Project", "Login - Collaborative Filtering", "Content Based Filtering"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:\n
                Trần Đăng Diệp & Vũ Thị Thanh Trúc""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                        Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 7/12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Vấn đề : Một khi đã hiểu được khách hàng và nhu cầu của họ, chúng ta xây dựng được 1 hệ thống gợi ý thông minh mang những sản phẩm phù hợp đến với khách hàng, từ đó nâng cao doanh số và trải nghiệm người dùng.
    """)  
    st.write("""###### => Yêu cầu: Xây dựng thuật toán đề xuất sản phẩm dựa trên lịch sử mua hàng.""")
    st.image('images/hasaki_banner_2.jpg', use_container_width=True)

elif choice == "Build Project":
    st.write("#### 1. Some data")
    st.write("##### san_pham.csv")
    st.dataframe(df_products.head(3))
    st.write("##### danh_gia.csv")
    st.dataframe(df_output.head(3))
    st.write("##### khach_hang.csv")
    st.dataframe(df_customer.head(3))

    st.write("#### 2. Visualize sentiment")
    fig1 = sns.countplot(data=df_products, x='diem_trung_binh')
    plt.xticks(rotation=90)    
    st.pyplot(fig1.figure)

    st.write("## I. Content Based Filtering")
    st.write('#### Data preprocessing')
    st.write(' -Tạo cột content, lọc khoảng trắng và lấy ra 200 từ của cột mo_ta ')
    st.write(' -Tạo cột content_wt , dùng word_tokenize của thư viện UndertheSea để tách từ')
    st.write(' Kết quả :')
    st.image('images/after_pre.png',width=900)

    
    st.write("####  Build model...")
    st.write("- Dùng TFIDF đề vector hóa nội dung cột content_wt")
    st.write("- Tính độ tương đồng bằng model Cosine")
    st.image("images/build_model_1.png",width=900)
    st.image("images/build_model_2.png",width=900)

    st.write("####  Evaluation")
    st.write("Thời gian tính toán : 0.08s ")
    st.write("Thơi gian tìm kiếm sản phẩm tương đồng : 0.001s ") 
    st.write('kết quả dự đoán :')
    st.image('images/after_build_1.png',width=900)
    st.image('images/after_build_model_1.png',width=900)

    #### Collaborative Filtering
    st.write("## II. Collaborative Filtering")
    st.write('#### Data preprocessing')
    st.write(' - Xóa dữ liệu Null, Nan ')
    st.write(' - Chọn ra các cột cần thiết để huấn luyện')
    st.image('images/build_Collab_1.png',width=900)
    st.write('- Kiểm tra kiểu dữ liệu')
    st.image('images/build_Collab_2.png',width=900)
    st.write('- Đọc dữ liệu bằng Reader ')
    st.image('images/build_Collab_reader.png',width=900)

    st.write("####  Build model and Evaluation")
    st.write("- Turning và lấy siêu tham số tốt nhất")
    st.write("- Huấn luyện mô hình với tham số đã turning")
    st.image("images/build_Collab_3.png",width=900)
    

    st.write("#### Prediction")
    st.write("""- Viết hàm gợi ý 5 sản phẩm tốt nhất cho khách hàng :
             Loại bỏ sản phẩm đã  mua , tính EstimateScore , 
             sắp xếp các giá trị theo thứ tự,
             lọc EstimateScore >= 4 """)
    st.image('images/build_Collab_4.png',width=900)

elif choice == 'Login - Collaborative Filtering':
    # lưu trạng thái đăng nhập
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    # lưu lại thông tin khách hàng đã đăng nhập
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if not st.session_state["logged_in"]:  
        st.title("Đăng nhập")
        user_options = [
            (row['username'], row['password'],row['ho_ten'])
            for _, row in df_customer[0:6].iterrows()]
        with st.container(border=True):
            selected_username = st.selectbox(
            "Chọn tài khoản",
            options=user_options,
            format_func=lambda x: x[0])
            selected_password = selected_username[1]
            password = st.text_input("Mật khẩu", type="password",value=selected_password)
            # Xử lý khi nhấn nút "Đăng nhập"
            # Gợi ý cho người dùng
            st.info("Mật khẩu tự động được điền theo tài khoản đã chọn.")
            if st.button("Đăng Nhập"):
                if selected_password == password:
                    st.session_state["logged_in"] = True
                    st.session_state["data"] = selected_username
                    st.rerun()
                else:
                    st.error("Sai tên đăng nhập hoặc mật khẩu.")

    
    else:
        selected_username = st.session_state["data"]
        st.success(f"Chào mừng {selected_username[2]} đã đăng nhập thành công.")
        if st.button("Đăng xuất"):
            
            st.session_state["logged_in"] = False
            st.rerun()
        Login_show(recommended_collab,display_recommended_products,df_customer,df_output,df_products,selected_username,algorithm_KNN)


   
        

elif choice == 'Content Based Filtering':
    # Step 2: Select a product and get similar product recommendations
    st.write("---")
    st.write("## Gợi ý sản phẩm tương tự")

    if 'random_products' not in st.session_state:
        st.session_state.random_products = df_products
    if 'select_products' not in st.session_state:
        st.session_state.select_products = None

    product_options = [
        (row['ten_san_pham'], row['ma_san_pham'])
        for _, row in st.session_state.random_products.iterrows()
    ]

    # Cập nhật sản phẩm mặc định trong selectbox
    if 'selected_ma_san_pham' in st.session_state:
        default_product = next(
            (option for option in product_options if option[1] == st.session_state.selected_ma_san_pham),
            product_options[0]
        )
    else:
        default_product = product_options[0]

    selected_product = st.selectbox(
    "Chọn sản phẩm",
    options=product_options,
    index=product_options.index(default_product),
    format_func=lambda x: x[0]
    )

    st.write("Bạn đã chọn:", selected_product)
    st.session_state.selected_ma_san_pham = selected_product[1]

    if st.session_state.selected_ma_san_pham:
        selected_product_df = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        if not selected_product_df.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product_df['ten_san_pham'].values[0])

            product_description = selected_product_df['mo_ta'].values[0]
            st.image(selected_product_df['hinh_anh'].values[0])
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')
            st.write('#### Giá :',selected_product_df['gia_ban'].values[0],' VNĐ')
            st.write('#### Sao :',selected_product_df['diem_trung_binh'].values[0],' ⭐')
            luotmua = df_output[df_output['ma_san_pham']==selected_product_df['ma_san_pham'].values[0]].groupby('ma_san_pham',as_index=False)['ma_khach_hang'].count()
            if len(luotmua) > 0:
                st.write("### Lượt mua: ", luotmua['ma_khach_hang'][0]," 🛒")
            else:
                st.write("### Lượt mua: 0 🛒")
            st.write("## Đánh giá ")
            #get product code
            msp = selected_product_df['ma_san_pham'].values[0]
            # Vẽ biểu đồ
            paint = df_output[df_output['ma_san_pham']==msp].groupby('so_sao',as_index=False)['ma_khach_hang'].count()
            paint.rename(columns={"ma_khach_hang":"tong_so_luong"}, inplace=True)
            
            if len(paint) >0 :
                with st.container(border=True):
                    st.write("Tổng số sao được vote từ khách hàng")
                    st.bar_chart(paint, x="so_sao", y="tong_so_luong", horizontal=True)
                      # lấy độ dài của df mã khách hàng
                df_mkh = df_output[df_output['ma_san_pham']==msp]
                
                # if len(df_mkh) >= 1:
                st.write("### Các bình luận của khách hàng ")
                for i in range(0,len(df_mkh)):
                    if i == 3:
                        break
                    with st.container(border=True):
                        # get mã khách hàng
                        mkh = df_mkh['ma_khach_hang'].values[i]
                        # Write
                        st.write("Khách hàng : ",df_customer[df_customer['ma_khach_hang']==mkh]['ho_ten'].values[0])
                        st.write("Bình Luận  : ",df_output[df_output['ma_san_pham']==msp]['noi_dung_binh_luan'].values[i])
                        st.write("Vote : ",df_output[df_output['ma_san_pham']==msp]['so_sao'].values[i],' ⭐')
                        st.write("Ngày bình luận : ",df_output[df_output['ma_san_pham']==msp]['ngay_binh_luan'].values[i])
                    
            else:
                st.write("### Sản chưa có lượt đánh giá nào!")

            st.write('### Các sản phẩm liên quan:')

            recommendations = get_recommendations(
                df_products,
                st.session_state.selected_ma_san_pham,
                cosine_sim=cosine_sim_new,
                nums=6
            )
            display_recommended_products_1(recommendations,st.session_state.selected_ma_san_pham,df_output, cols=3)
            st.success('Đã đổi thông tin sản phẩm ')   



        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
            


