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

# @st.cache_resource
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

# 1. Configuración de la página (Ícono en la pestaña del navegador)
st.set_page_config(
                   page_title="Dashboard Disdrómetro SMN", 
                   page_icon="disdro_icon.png", 
                   layout="wide"
                  )
st.image("smn_horizontal_arg-01.jpg", width=250)

# 2. Título principal con ícono alineado
col_icon, col_title = st.columns([1, 15]) # Proporción para que el ícono ocupe poco espacio

with col_icon:
    # Mostramos el ícono con un ancho fijo en píxeles
    st.image("disdro_icon.png", width=65)

with col_title:
    # Ajustamos un poco el margen superior usando markdown/HTML para alinearlo perfectamente con el centro del ícono
    st.markdown('<h1 style="margin-top: -12px;">Disdrómetro SMN | Dashboard Interactivo</h1>', unsafe_allow_html=True)
    st.markdown(
    '<p style="font-size: 20px; ">Dirección de Productos de Modelación Ambiental y de Sensores Remotos - DNCIPS</p>', 
    unsafe_allow_html=True)

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
    df_ubicacion = pd.DataFrame({'lat': [-34.56405], 'lon': [-58.41742]})
    st.map(df_ubicacion, zoom=13, use_container_width=True)

st.markdown("---")
# --- FIN DEL ENCABEZADO ---

logo_path = Path('./smn_logo.png')
smn_logo = mpimg.imread(logo_path) if logo_path.exists() else None

# Al quitar el botón, esto se ejecuta automáticamente apenas carga la página
with st.spinner("Obteniendo datos históricos..."):
    try:
        drive_service = get_drive_service()
        # Leemos el archivo compilado desde Drive
        df_resumen = buscar_y_leer_csv(drive_service, 'resumen_historico.csv')
        
        if df_resumen is not None:
            # Generamos la imagen
            path_out_resumen = Path('./imagenes/resumen/')
            path_out_resumen.mkdir(parents=True, exist_ok=True)
            
            utils_disdro.plot_resumen_historico(
                df=df_resumen, 
                path_out=path_out_resumen, 
                smn_logo=smn_logo
            )
            
            # Desplegamos la imagen
            archivo_resumen = path_out_resumen.joinpath('disdro_SMN-Dorrego_resumen_historico.png')
            if archivo_resumen.exists():
                st.image(str(archivo_resumen), use_container_width=True)
                
                # Leyenda rápida debajo del gráfico
                st.caption("🟢 Dato disponible | 🔴 Sin registros/Equipo apagado")
        else:
            st.warning("No se encontró el archivo 'resumen_historico.csv' en Google Drive.")
    except Exception as e:
        st.error(f"Error procesando el histórico: {e}")

path_out2  = Path('./imagenes/daily/hietograma/')
path_out3  = Path('./imagenes/daily/scatter_Dm_Nw/')
path_out4  = Path('./imagenes/daily/scatter_Z_R/')

for p in [path_out2, path_out3, path_out4]:
    p.mkdir(parents=True, exist_ok=True)

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