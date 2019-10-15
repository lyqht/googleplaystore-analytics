echo "\
[general]\n\
email=\"estee.tsw@gmail.com\"\n\
" >~/.streamlit/credentials.toml

echo "\
[server]\n]
enableCORS=false
headless = true\n\
port = $PORT\n\
" > ~/.streamlit.config.toml