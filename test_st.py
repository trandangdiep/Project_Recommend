import streamlit as st

# Khởi tạo giỏ hàng nếu chưa có
if "cart" not in st.session_state:
    st.session_state["cart"] = []

# Hàm hiển thị giỏ hàng
def display_cart():
    st.write("### Giỏ hàng của bạn:")
    if st.session_state["cart"]:
        for i, item in enumerate(st.session_state["cart"], 1):
            st.write(f"{i}. {item}")
    else:
        st.write("Giỏ hàng hiện đang trống.")
    if st.button("Xóa giỏ hàng"):
        st.session_state["cart"] = []
        st.success("Giỏ hàng đã được xóa.")

# Hàm thêm sản phẩm vào giỏ hàng
def add_to_cart(item):
    st.session_state["cart"].append(item)
    st.success(f"'{item}' đã được thêm vào giỏ hàng.")

# Giao diện chính
st.title("Ứng dụng Giỏ hàng")

# Danh sách sản phẩm
products = ["Sản phẩm A", "Sản phẩm B", "Sản phẩm C", "Sản phẩm D"]
st.write("### Danh sách sản phẩm:")
for product in products:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(product)
    with col2:
        if st.button(f"Thêm {product}", key=product):
            add_to_cart(product)

st.divider()

# Hiển thị giỏ hàng
display_cart()