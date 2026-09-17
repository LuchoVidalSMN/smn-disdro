#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:36:04 2026
@author: Dr. Luciano Vidal DPMAYSR-DNCIPS-SMN
"""

import io
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
import matplotlib.image as mpimg
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import utils_disdro

# --- CONFIGURACIÓN GOOGLE DRIVE ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

@st.cache_resource
def get_drive_service():
    """Autentica y crea el servicio de Google Drive usando los secretos de Streamlit."""
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def buscar_y_leer_csv(service, nombre_archivo):
    """Busca un archivo por nombre en Drive y lo lee a un DataFrame."""
    # Buscamos el archivo por nombre
    query = f"name='{nombre_archivo}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        return None
    
    # Tomamos el ID del primer archivo que coincida
    file_id = items[0]['id']
    
    # Descargamos el contenido en memoria
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        
    fh.seek(0)
    # Leemos el CSV con Pandas
    df = pd.read_csv(fh)
    return df

# --- RESTO DEL DASHBOARD ---
st.set_page_config(page_title="Dashboard Disdrómetro SMN", layout="wide")
st.title("🌧️ Disdrómetro SMN-Dorrego | Dashboard Interactivo (EXPERIMENTAL)")

# --- ENCABEZADO DEL INSTRUMENTO ---
st.markdown("---")
st.subheader("Acerca del Instrumento")

# Dividimos la pantalla en 3 columnas (proporciones: más ancha para texto, iguales para foto y mapa)
col_texto, col_img, col_mapa = st.columns([2, 1.2, 1.2])

with col_texto:
    st.markdown("""
    **Modelo:** OTT Parsivel² (Disdrómetro Óptico Láser)  
    **Ubicación:** [Estación Meteorológica Dorrego - SMN](https://maps.app.goo.gl/7PPE2mYATrTNpQ9x8)
    
    **Principio de funcionamiento:**  
    El equipo emite un haz de luz láser horizontal continuo entre sus dos cabezales. Cuando la precipitación atraviesa este haz, interrumpe temporalmente una fracción de la señal óptica.
    
    * **Tamaño de la gota:** Se calcula analizando la amplitud máxima del oscurecimiento (cuánta luz bloquea la partícula).
    * **Velocidad de caída:** Se determina midiendo el tiempo que la partícula tarda en cruzar por completo el haz de luz.
    
    Combinando el tamaño y la velocidad individual de miles de gotas, el algoritmo reconstruye la microfísica de la tormenta, permitiendo derivar la tasa de lluvia, el equivalente de reflectividad radar (Z) y la energía cinética.
    """)

with col_img:
    try:
        # Se asegura de leer la imagen si está en el mismo directorio
        st.image("20250428_132821.jpg", caption="OTT Parsivel² instalado en el sitio", use_container_width=True)
    except FileNotFoundError:
        st.info("Imagen del sitio no encontrada.")

with col_mapa:
    # Coordenadas de la zona de la Estación Dorrego (CABA) para el st.map
    # Puedes ajustar los decimales si quieres correr el punto exacto
    df_ubicacion = pd.DataFrame({'lat': [-34.5663], 'lon': [-58.4285]})
    st.map(df_ubicacion, zoom=13, use_container_width=True)

st.markdown("---")
# --- FIN DEL ENCABEZADO ---

path_out2  = Path('./imagenes/daily/hietograma/')
path_out3  = Path('./imagenes/daily/scatter_Dm_Nw/')
path_out4  = Path('./imagenes/daily/scatter_Z_R/')

for p in [path_out2, path_out3, path_out4]:
    p.mkdir(parents=True, exist_ok=True)

logo_path = Path('./smn_logo.png')
smn_logo = mpimg.imread(logo_path) if logo_path.exists() else None

st.sidebar.header("Parámetros de Visualización")
fecha_seleccionada = st.sidebar.date_input("Seleccionar Fecha", datetime.today())
fecha_str = fecha_seleccionada.strftime('%Y%m%d')
plot_date = fecha_seleccionada.strftime('%Y-%m-%d')
nombre_csv_esperado = f'salida_disdrometro_{fecha_str}.csv'

if st.sidebar.button("Generar Figuras"):
    with st.spinner("Conectando a Google Drive..."):
        try:
            drive_service = get_drive_service()
            df = buscar_y_leer_csv(drive_service, nombre_csv_esperado)
            
            if df is not None:
                st.success(f"Datos cargados correctamente: `{nombre_csv_esperado}`")
                
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                fecha_obj = datetime.combine(fecha_seleccionada, datetime.min.time())
                hubo_lluvia = df['Lluvia_Acumulada_Total_mm'].iloc[-1] > 0.0

                file_hietograma = path_out2.joinpath(f'disdro_SMN-Dorrego_serie_temporal_{plot_date}.png')
                file_zr = path_out4.joinpath(f'disdro_SMN-Dorrego_parametros_Z_R_{fecha_str}.png')
                file_dm_nw = path_out3.joinpath(f'disdro_SMN-Dorrego_Dm_vs_log10Nw_{fecha_str}.png')

                with st.spinner("Generando figuras..."):
                    if hubo_lluvia:
                        utils_disdro.plot_hietograma(df=df, fecha_obj=fecha_obj, plot_date=plot_date, path_out2=path_out2, smn_logo=smn_logo)
                        utils_disdro.plot_zr(df=df, fecha=fecha_obj, path_imagenes=path_out4, smn_logo=smn_logo)
                        utils_disdro.plot_dm_vs_log10nw(df=df, path_imagenes=path_out3, smn_logo=smn_logo)
                    else:
                        utils_disdro.plot_hietograma_sin_datos(df=df, fecha_obj=fecha_obj, plot_date=plot_date, path_out2=path_out2, smn_logo=smn_logo)
                        utils_disdro.plot_zr_sin_datos(fecha=fecha_obj, path_imagenes=path_out4, smn_logo=smn_logo)
                        utils_disdro.plot_dm_vs_log10nw_sin_datos(fecha_obj=fecha_obj, path_imagenes=path_out3, smn_logo=smn_logo)

                # Despliegue de imágenes (como configuramos antes)
                st.markdown("---")
                # st.subheader("Hietograma Diario")
                if file_hietograma.exists():
                    st.image(str(file_hietograma), use_container_width=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    # st.subheader("Parámetros Z-R")
                    if file_zr.exists():
                        st.image(str(file_zr), use_container_width=True)
                with col2:
                    # st.subheader("Microfísica (Dm vs log10 Nw)")
                    if file_dm_nw.exists():
                        st.image(str(file_dm_nw), use_container_width=True)
            else:
                st.error(f"El archivo `{nombre_csv_esperado}` no se encontró en la carpeta compartida de Google Drive.")
        except Exception as e:
            st.error(f"Error al conectar con Google Drive: {e}")