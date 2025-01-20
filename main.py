'''
    Author: Miliano Fernandes de Oliveira Junior
    Date: 03/02/2024
    Description: Este é um projeto que vai estar no github e será vinculado ao servidor da railway, sendo assim, o
    projeto será um dashboard que vai mostrar os dados de um banco de dados que será alimentado por varias tabelas
    que já estão no servidor da railway.


    1 passo: Criar um projeto no github e vincular ao projeto local. ->OK
    2 passo: Vincular o repositório do github ao servidor da railway. ->OK
    3 passo: Configurar o servidor da railway para rodar o projeto. -> OK
'''

from libs.new_analise import new_analise
import streamlit as st

# load_dotenv()
def page_principal():
    ''' Página principal do dashboard que cotém as comparações entre as usinas'''
    new_analise('CGH Aparecida')
        

def pages():
    ''' Header do dashboard '''
    # carregamento do css
    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


    page_principal()

def main():
    ''' Dashboard principal '''
    # configuração da página
    st.set_page_config(
                        layout="wide",
                        page_title="EngeSEP",
                        page_icon="📊",
                        initial_sidebar_state="expanded",
                     )
    # Define estilos para os subcabeçalhos no padrão ABNT
    subheader_style = """
        <style>
            .subheader-abnt {
                font-size: 12px; /* Tamanho da fonte ABNT para seções secundárias */
                font-weight: bold;
            }
        </style>
    """
    st.markdown(subheader_style, unsafe_allow_html=True)



    st.markdown(
    """
    <style>
        .subheader-abnt {
            font-size: 16px;
            font-weight: bold;
        }

        @media print {
            body {
                font-size: 10pt;
                -webkit-print-color-adjust: exact !important;
                color-adjust: exact !important;
            }
            .subheader-abnt {
                font-size: 11px;
            }
            h1, h2, h3, h4, h5, h6 {
                break-after: avoid;
            }
            .stDataFrame {
                overflow-x: auto !important; /* Prevent horizontal overflow */
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    # # Header
    pages()



if __name__ == '__main__':

    main()


