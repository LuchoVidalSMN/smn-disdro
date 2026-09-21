#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified on Mon Jul 6, 2026

@author: Dr. Luciano Vidal DPMAYSR-DNCIPS-SMN

"""
# =========================================================================== #

import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para trabajar sin interfaz gráfica

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

# =========================================================================== #

bins = np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812,
                    0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125,
                    2.375, 2.75, 3.25, 3.75, 4.25, 4.75, 5.5, 6.5, 7.5,
                    8.5, 9.5, 11., 13., 15., 17., 19., 21.5, 24.5])

bins_diff = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                0.125, 0.125, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5,
                0.5, 0.5, 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3.]

drop_vel = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                        0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3,
                        3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12,
                        13.6, 15.2, 17.6, 20.8])

# Velocidad del paper de Campos 2006
vf = -0.193 + (4.96 * bins) - (0.904 * np.power(bins, 2)) + \
    (0.0566 * np.power(bins, 3))

# =========================================================================== #

mascara = np.zeros((32, 32), dtype='int')

# Esta mascara viene del paper de Campos 2006
# 60% de la velocidad calculada mas arriba
matriz_velocidad = np.reshape(np.repeat(drop_vel, 32), (32, 32))
for i in range(vf.size) :
    for j in range(vf.size) :
        if not(vf[j]*0.4 < matriz_velocidad[i][j] < vf[j]*1.6) :
            mascara[i][j] = 1

# =========================================================================== #

def leo_espectro(archivo) :

    fecha = datetime.strptime(archivo.stem, 'Datos -- %d%m%Y - %H%M%S')

    with open(archivo, 'r') as f :
        lines = f.readlines()

    try :
        # Moved this line into the try block to catch IndexError from malformed files
        espectro_raw = np.asarray(lines[1].split(':')[1].split(';')[:-1], dtype='int')
        matriz = espectro_raw.reshape(32, 32)
    except (ValueError, IndexError) as e :
        # Return None for all expected outputs if an error occurs
        # Added None for Dm, Nw, log10_Nw
        return None, None, None, None, None, None, None, None, None, None, None

    matriz_mask = np.ma.array(matriz, mask=mascara)

    NDS = np.sum(matriz_mask, axis=0)

    # Intervalo de medición del disdrometro en segundos
    T = 60

    # Área de medición efectiva.
    # El láser mide 180 mm x 30 mm = 5400 mm^2 = 0.0054 m^2.
    SUP = 180. * 30. * np.power(10., -6.)

    DG = NDS / (vf * SUP * T * bins_diff)
    DG = np.around(DG, 4)

    # Factor de Reflectividad del Radar [dBZ]
    ray_Z = np.sum(np.multiply(DG*bins_diff, np.power(bins, 6.)))
    dBZ = 10. * np.log10(ray_Z) if ray_Z > 0 else -99.9

    # Tasa Instantánea de Lluvia [mm/h]
    rr_1 = DG * vf * (np.pi/6) * 3.6e-3
    RR = np.sum(np.multiply(rr_1*bins_diff, np.power(bins, 3.))) if np.sum(DG)>0 else 0.0

    # Lluvia Acumulada en el intervalo T (mm)
    AccRain = RR * (T / 3600) if RR > 0 else 0.0

    # Energía cinética de cada gota (Joules) calculada vectorialmente
    e_gota = (np.pi / 12.) * 1e-6 * np.power(bins, 3.) * np.power(vf, 2.)

    # Energía cinética temporal en el intervalo (J / m^2)
    KE_time = np.sum((NDS / SUP) * e_gota) if np.sum(DG)>0 else 0.0

    # Energía cinética por milímetro de lluvia (J / (m^2 * mm))
    KE_mm = KE_time / AccRain if AccRain > 0 else 0.0

    # Diámetro medio ponderado por masa (Dm en mm)
    # Dm = (Sum(N(Di) * Di^4 * dDi)) / (Sum(N(Di) * Di^3 * dDi))
    numerator_Dm = np.sum(DG * np.power(bins, 4.) * bins_diff)
    denominator_Dm = np.sum(DG * np.power(bins, 3.) * bins_diff)

    Dm = numerator_Dm / denominator_Dm if denominator_Dm > 0 else 0.0

    # Parámetro de intercepto normalizado (Nw en mm^-1 m^-3)
    # Nw = (128/3) * 10^-3 * (M3 / Dm^4)
    # Where M3 = sum(DG * bins^3 * bins_diff) in mm^3 m^-3
    # This derivation assumes rho_w = 1 g/cm^3 = 10^-3 g/mm^3 in LWC calculation
    if Dm > 0 :
        Nw = (128 / 3) * 1e-3 * (denominator_Dm / np.power(Dm, 4.))
    else :
        Nw = 0.0

    log10_Nw = np.log10(Nw) if Nw > 0 else -99.9 # Use -99.9 for log of zero/negative values

    return fecha, dBZ, RR, AccRain, KE_time, KE_mm, DG, matriz_mask, Dm, Nw, log10_Nw # Added Dm, Nw, log10_Nw

# =========================================================================== #

def plot_hietograma(df, fecha_obj, plot_date, path_out2, smn_logo=None) :
    
    """
    Genera y guarda el gráfico de tasa de lluvia y lluvia acumulada para un día específico.
    
    Parámetros:
    - df: DataFrame con los datos del día.
    - fecha_obj: Objeto datetime para formatear la fecha en el título y nombre de archivo.
    - plot_date: String de la fecha en formato 'YYYY-MM-DD' para los límites del eje X.
    - path_out2: Objeto Path (pathlib) donde se guardará la imagen.
    - smn_logo: Imagen del logo cargada (opcional).
    """
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    try :
        
        # Plot Tasa_Lluvia_mm_h on the primary y-axis (left) as a bar plot using ax1.bar
        ax1.bar(
            x=df['Fecha'],
            height=df['Tasa_Lluvia_mm_h'],
            color='blue',
            alpha=0.75,
            width=timedelta(minutes=1)*1.0, # Adjust width as needed for 1-minute intervals
            label='_nolegend_'
        )
        ax1.set_xlabel('Hora Local', fontsize=12)
        ax1.set_ylabel('Tasa de Lluvia [mm/h]', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(bottom=0) # Set primary y-axis to start at 0
        
        # Create a second y-axis for Lluvia_Acumulada_Total_mm
        ax2 = ax1.twinx()
        sns.lineplot(
            x='Fecha',
            y='Lluvia_Acumulada_Total_mm',
            data=df,
            color='red',
            ax=ax2,
            label='_nolegend_',
            linewidth=2.0
        )
        ax2.set_ylabel('Lluvia Acumulada [mm]', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(bottom=0) # Set secondary y-axis to start at 0
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(10)) # Space labels every 10 mm
        
        # Calculate total accumulated rain for the day to include in the title
        total_accumulated_rain = df['Lluvia_Acumulada_Total_mm'].iloc[-1] if not df.empty else 0
        
        # Set title with the date and total accumulated rain
        plt.title(f'Disdrómetro SMN-Dorrego | {plot_date}\nTotal: {total_accumulated_rain:.2f} mm',
                  fontsize=16,
                  fontweight='bold')
        
        # Format x-axis to show only hours, every 1 hour
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        
        # Set x-axis limits from 00:00 to 00:00 the next day
        start_time = datetime.strptime(plot_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        end_time = start_time + timedelta(days=1)
        ax1.set_xlim(start_time, end_time)
        
        # Manually create the legend using the line artists and custom labels
        line1_artist = ax1.patches[0] if ax1.patches else None # Get a patch from barplot for legend
        line2_artist = ax2.lines[0] if ax2.lines else None
        
        # Handle case where no bars might be plotted if data is empty
        if line1_artist is not None and line2_artist is not None :
            ax1.legend([line1_artist, line2_artist], ['Tasa de Lluvia', 'Lluvia Acumulada'], loc='upper left')
        elif line2_artist is not None :
            ax1.legend([line2_artist], ['Lluvia Acumulada'], loc='upper left')
        
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        # Agregar el logo como marca de agua si se cargó correctamente
        if smn_logo is not None :
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1) 
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='figure fraction', frameon=False, pad=0.0)
            fig.add_artist(ab)
        
        # Add copyright notice
        fig.text(0.83, 0.03, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')
        
        # Save the figure as a PNG file
        output_filename = f'disdro_SMN-Dorrego_serie_temporal_{plot_date}.png'
        output_path = path_out2.joinpath(output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {output_path}")

    finally :
        
        # Esto se ejecuta SIEMPRE, incluso si el código de arriba falla, evitando fugas de memoria
        plt.close(fig)

# =========================================================================== #

def plot_hietograma_sin_datos(df, fecha_obj, plot_date, path_out2, smn_logo=None) :
    
    """
    Genera y guarda la misma estructura de gráfico pero vacía, indicando "Sin Datos" o "No Data" 
    en el centro para los días en los que no hubo registros de lluvia.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    try :
        
        # Configuración estética del eje principal (Izquierdo)
        ax1.set_xlabel('Hora Local', fontsize=12)
        ax1.set_ylabel('Tasa de Lluvia [mm/h]', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 10) # Un límite fijo por defecto ya que no hay datos
        
        # Configuración estética del eje secundario (Derecho)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Lluvia Acumulada [mm]', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 10)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(2))
        
        # Texto central "No Data"
        # Usamos ax1.transAxes para que (0.5, 0.5) sea exactamente el centro del recuadro del gráfico
        ax1.text(0.5, 0.5, 'No Data / Sin Registros', 
                 fontsize=20, 
                 fontweight='bold', 
                 color='gray', 
                 alpha=0.6,
                 ha='center', 
                 va='center', 
                 transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10))
        
        # Título del gráfico (Total siempre será 0.00 mm)
        plt.title(f'Disdrómetro SMN-Dorrego | {plot_date}\nTotal: 0.00 mm',
                  fontsize=16,
                  fontweight='bold')
        
        # Formateo del eje X de tiempo (Igual al gráfico original)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Límites del eje X
        start_time = datetime.strptime(plot_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        end_time = start_time + timedelta(days=1)
        ax1.set_xlim(start_time, end_time)
        
        plt.tight_layout()
        
        # Agregar el logo como marca de agua si está disponible
        if smn_logo is not None :
            
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1) 
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='figure fraction', frameon=False, pad=0.0)
            fig.add_artist(ab)
        
        # Add copyright notice
        fig.text(0.83, 0.03, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')
                
        # Guardar la figura
        output_filename = f'disdro_SMN-Dorrego_serie_temporal_{plot_date}.png'
        output_path = path_out2.joinpath(output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figura de día sin datos guardada en: {output_path}")

    finally :
        
        plt.close(fig)

# =========================================================================== #

def plot_zr(df, fecha, path_imagenes, smn_logo=None) :
    
    """
    Filtra los datos, realiza la regresión lineal para obtener los parámetros Z-R
    y genera/guarda el gráfico de dispersión con la línea de ajuste y métricas.
    
    Parámetros:
    - df: DataFrame con los datos brutos de Reflectividad y Tasa de Lluvia.
    - fecha: Objeto datetime para el título y nombre de archivo.
    - path_imagenes: Objeto Path (pathlib) para guardar el resultado.
    - smn_logo: Imagen del logo cargada (opcional).
    """
    # 1. Filtrado de datos
    filtered_df = df[df['Reflectividad_dBZ'] != -99.9].copy()
    filtered_df = filtered_df[filtered_df['Tasa_Lluvia_mm_h'] > 0].copy()
    
    # Validación de seguridad: se necesitan al menos 2 puntos para tirar una recta
    if len(filtered_df) < 2:
        print(f"[{fecha.strftime('%d/%m/%Y')}] No hay suficientes datos válidos para calcular Z-R (puntos: {len(filtered_df)}).")
        return None

    # 2. Cálculo del logaritmo y regresión lineal
    filtered_df['log10_Tasa_Lluvia_mm_h'] = np.log10(filtered_df['Tasa_Lluvia_mm_h'])
    
    slope, intercept, r_value, p_value, std_err = linregress(
        filtered_df['log10_Tasa_Lluvia_mm_h'], 
        filtered_df['Reflectividad_dBZ']
    )
    
    # 3. Cálculo de predicción y RMSE
    predicted_dBZ = slope * filtered_df['log10_Tasa_Lluvia_mm_h'] + intercept
    rmse = np.sqrt(mean_squared_error(filtered_df['Reflectividad_dBZ'], predicted_dBZ))
    
    # 4. Generación del gráfico
    fig, ax = plt.subplots(figsize=(6, 6))
    
    try:
        sns.scatterplot(x='log10_Tasa_Lluvia_mm_h', y='Reflectividad_dBZ', data=filtered_df, alpha=0.6, ax=ax)
        sns.regplot(x='log10_Tasa_Lluvia_mm_h', y='Reflectividad_dBZ', data=filtered_df, scatter=False, color='red', line_kws={'linestyle':'--'}, ax=ax)
        
        # Marca de agua del logo
        if smn_logo is not None:
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='figure fraction', frameon=False, pad=0.0)
            fig.add_artist(ab)
            
        # Cuadros de texto con resultados de la regresión
        # Nota: 'a' y 'b' se calculan según las conversiones de unidades estándar en radar/disdrómetros
        text_kwargs = dict(transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.5))
        
        ax.text(0.04, 0.95, f'a: {np.power(10, intercept/10):.2f}', **{**text_kwargs, 'fontsize': 14})
        ax.text(0.04, 0.88, f'b: {slope/10:.2f}', **{**text_kwargs, 'fontsize': 14})
        ax.text(0.04, 0.81, f'R: {r_value:.2f}', **text_kwargs)
        ax.text(0.04, 0.74, f'R^2: {r_value**2:.2f}', **text_kwargs)
        ax.text(0.04, 0.67, f'RMSE: {rmse:.2f}', **text_kwargs)
        
        # Títulos y etiquetas de los ejes
        plt.title(f'Disdrómetro SMN-Dorrego | {fecha.strftime("%d/%m/%Y")}\nParámetros Z-R', fontsize=16, fontweight='bold')
        plt.xlabel(r'$log_{10}(RR)$ [mm/h]', fontsize=12)
        plt.ylabel('Reflectividad [dBZ]', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        # Nota de Copyright fija al 2026
        fig.text(0.78, 0.12, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')
        
        # Guardado del archivo
        output_filename = f'disdro_SMN-Dorrego_parametros_Z_R_{fecha.strftime("%Y%m%d")}.png'
        output_path = path_imagenes.joinpath(output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {output_path}")
        
    finally:
        plt.close(fig)

# =========================================================================== #

def plot_zr_sin_datos(fecha, path_imagenes, smn_logo=None) :
    
    """
    Genera y guarda la misma estructura del gráfico de parámetros Z-R vacío, 
    indicando "No Data" en el centro para los días en los que no hubo lluvias.
    
    Parámetros:
    - fecha: Objeto datetime para el título y nombre de archivo.
    - path_imagenes: Objeto Path (pathlib) para guardar el resultado.
    - smn_logo: Imagen del logo cargada (opcional).
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    try:
        # Configuración estética de los ejes (idéntica al gráfico original)
        plt.xlabel(r'$log_{10}(RR)$ [mm/h]', fontsize=12)
        plt.ylabel('Reflectividad [dBZ]', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Fijamos límites lógicos para que los ejes no se autoescalen a (0,1)
        ax.set_xlim(-1, 2)
        ax.set_ylim(0, 60)
        
        # Texto central "No Data"
        # (0.5, 0.5) en ax.transAxes es exactamente el centro del recuadro del gráfico
        ax.text(0.5, 0.5, 'No Data / Sin Registros', 
                 fontsize=18, 
                 fontweight='bold', 
                 color='gray', 
                 alpha=0.6,
                 ha='center', 
                 va='center', 
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10))
        
        # Título del gráfico 
        plt.title(f'Disdrómetro SMN-Dorrego | {fecha.strftime("%d/%m/%Y")}\nParámetros Z-R', 
                  fontsize=16, 
                  fontweight='bold')
        
        # Marca de agua del logo si está disponible
        if smn_logo is not None:
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1)
            ab = AnnotationBbox(imagebox, (0.53, 0.485), xycoords='figure fraction', frameon=False, pad=0.0)
            fig.add_artist(ab)
            
        plt.tight_layout()
        
        # Nota de Copyright fija
        fig.text(0.75, 0.12, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')
        
        # Guardado del archivo con un sufijo aclaratorio
        output_filename = f'disdro_SMN-Dorrego_parametros_Z_R_{fecha.strftime("%Y%m%d")}.png'
        output_path = path_imagenes.joinpath(output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figura Z-R de día sin datos guardada en: {output_path}")
        
    finally:
        plt.close(fig)
        
# =========================================================================== #

def plot_dm_vs_log10nw(df, path_imagenes, smn_logo=None) :
    
    """
    Filtra los datos microfísicos y genera un gráfico de dispersión (Scatter Plot)
    de Dm vs log10(Nw) mapeando el tamaño y color de los puntos según la Tasa de Lluvia.
    
    Parámetros:
    - df: DataFrame con los datos del día (debe contener 'Fecha', 'log10_Nw', 'Dm_mm' y 'Tasa_Lluvia_mm_h').
    - path_imagenes: Objeto Path (pathlib) donde se guardará la imagen.
    - smn_logo: Imagen del logo cargada (opcional).
    """

    # Asegurar tipo datetime en la columna Fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Obtener la fecha para el título y nombre de archivo usando la primera fila válida
    plot_date_str  = df['Fecha'].iloc[0].strftime('%Y-%m-%d')
    plot_date_file = df['Fecha'].iloc[0].strftime('%Y%m%d')

    # Filtrado de valores inválidos o nulos
    filtered_plot_df = df[
                          ( df['log10_Nw'] != -99.9 ) &
                          ( df['Dm_mm'] > 0 )
                         ].copy()

    # Validación de seguridad: si no hay puntos tras el filtrado, salir limpiamente
    if filtered_plot_df.empty :
        print(f"[{plot_date_str}] Sin datos válidos de Dm y log10(Nw) para graficar microfísica.")
        return None

    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    try:
        # Gráfico de dispersión multivariable
        sns.scatterplot(
            x='Dm_mm',
            y='log10_Nw',
            data=filtered_plot_df,
            hue='Tasa_Lluvia_mm_h',
            size='Tasa_Lluvia_mm_h',
            sizes=(20, 400),
            palette='viridis',
            alpha=0.7,
            ax=ax_scatter
        )

        # Configuración de etiquetas y grilla
        ax_scatter.set_xlabel('Diámetro Medio Ponderado por Masa (Dm) [mm]', fontsize=12)
        ax_scatter.set_ylabel(r'$\log_{10}(N_w)$ $[mm^{-1} m^{-3}]$', fontsize=12)
        ax_scatter.grid(True, linestyle='--', alpha=0.7)

        # Título dinámico
        plt.title(f'Disdrómetro SMN-Dorrego | {plot_date_str}\n$D_m$ vs $\\log_{{10}}N_w$ (coloreado por Tasa de Lluvia)',
                  fontsize=16,
                  fontweight='bold')

        # Nota de copyright fija al año 2026
        fig_scatter.text(0.87, 0.11, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')

        # Marca de agua del SMN
        if smn_logo is not None:
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='figure fraction', frameon=False, pad=0.0)
            fig_scatter.add_artist(ab)

        plt.tight_layout()

        # Guardado de la imagen
        output_filename_scatter = f'disdro_SMN-Dorrego_Dm_vs_log10Nw_{plot_date_file}.png'
        output_path = path_imagenes.joinpath(output_filename_scatter)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figura $D_m$ vs $\\log_{{10}}N_w$ guardada en: {output_path}")

    finally:
        plt.close(fig_scatter)

# =========================================================================== #

def plot_dm_vs_log10nw_sin_datos(fecha_obj, path_imagenes, smn_logo=None) :
    
    """
    Genera y guarda la misma estructura del gráfico microfísico pero vacío, 
    colocando un cartel "No Data" central para los días secos o sin registros.
    
    Parámetros:
    - fecha_obj: Objeto datetime para extraer la fecha del día.
    - path_imagenes: Objeto Path (pathlib) donde se guardará la imagen.
    - smn_logo: Imagen del logo cargada (opcional).
    """
    
    plot_date_str = fecha_obj.strftime('%Y-%m-%d')
    plot_date_file = fecha_obj.strftime('%Y%m%d')

    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    try :
        
        # Definición de límites fijos estándar para el scatter para que conserve proporciones realistas
        ax_scatter.set_xlim(0, 6)
        ax_scatter.set_ylim(0, 8)
        
        ax_scatter.set_xlabel('Diámetro Medio Ponderado por Masa (Dm) [mm]', fontsize=12)
        ax_scatter.set_ylabel(r'$\log_{10}(N_w)$ $[mm^{-1} m^{-3}]$', fontsize=12)
        ax_scatter.grid(True, linestyle='--', alpha=0.7)

        # Cartel central de No Data
        ax_scatter.text(0.5, 0.5, 'No Data / Sin Registros', 
                        fontsize=20, 
                        fontweight='bold', 
                        color='gray', 
                        alpha=0.6,
                        ha='center', 
                        va='center', 
                        transform=ax_scatter.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=12))

        # Título institucional
        plt.title(f'Disdrómetro SMN-Dorrego | {plot_date_str}\n$D_m$ vs $\\log_{{10}}N_w$ (coloreado por Tasa de Lluvia)',
                  fontsize=16,
                  fontweight='bold')

        # Copyright fijo
        fig_scatter.text(0.87, 0.11, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')

        # Marca de agua del logo
        if smn_logo is not None:
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1)
            ab = AnnotationBbox(imagebox, (0.515, 0.5), xycoords='figure fraction', frameon=False, pad=0.0)
            fig_scatter.add_artist(ab)

        plt.tight_layout()

        # Guardar archivo vacío
        output_filename_scatter = f'disdro_SMN-Dorrego_Dm_vs_log10Nw_{plot_date_file}.png'
        output_path = path_imagenes.joinpath(output_filename_scatter)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura microfísica de día sin datos guardada en: {output_path}")

    finally:
        plt.close(fig_scatter)

# =========================================================================== #

def plot_espectrograma_dsd(df, clases_diametros, matriz_nd, fecha_obj, plot_date, path_out, smn_logo=None) :
    
    """
    Genera un espectrograma temporal de la distribución de tamaño de gotas N(D) vs Diámetro y Tiempo.
    
    Parámetros:
    - df: DataFrame original (usado para extraer la columna 'Fecha').
    - clases_diametros: Lista o array con los centros de los canales de diámetros [mm] (eje Y).
    - matriz_nd: Matriz bidimensional (Numpy array) de dimensiones (len(clases_diametros), len(df)) 
                 que contiene los valores de N(D) [m^-3 mm^-1].
    - fecha_obj: Objeto datetime para títulos y nombres de archivos.
    - plot_date: String 'YYYY-MM-DD' para los límites del eje X.
    - path_out: Objeto Path (pathlib) para guardar la imagen.
    - smn_logo: Logo institucional (opcional).
    """
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    try :
        
        # Aseguramos formato datetime en el tiempo
        tiempos = pd.to_datetime(df['Fecha'])
        
        # Reemplazar ceros o valores negativos por NaN para que no pinten en el pcolormesh (queden blancos)
        matriz_plot = np.where(matriz_nd <= 0, np.nan, matriz_nd)
        
        # Graficar usando pcolormesh (X: Tiempos, Y: Diámetros, Z: Concentración N(D))
        # Shading='nearest' centra el color en las coordenadas dadas
        pcm = ax.pcolormesh(tiempos, clases_diametros, matriz_plot, cmap='jet', vmin=0, vmax=1200, shading='nearest')
        
        # Configuración de la barra de colores (Colorbar)
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, extend='max')
        cbar.set_label(r'$N(D)$ $[m^{-3} mm^{-1}]$', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Configuración de ejes
        ax.set_xlabel('Hora Local', fontsize=12)
        ax.set_ylabel('Drop diameter [mm]', fontsize=12)
        ax.set_ylim(0.3, 3.7) # Ajustado al rango típico del gráfico que mostraste
        
        # Formateo estricto del eje X de tiempo (De 00:00 a 24:00 del día correspondiente)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        start_time = pd.to_datetime(plot_date + ' 00:00:00')
        end_time = start_time + pd.Timedelta(days=1)
        ax.set_xlim(start_time, end_time)
        
        # Título institucional
        plt.title(f'Disdrómetro SMN-Dorrego | {fecha_obj.strftime("%d/%m/%Y")}\nEspectrograma Temporal de Gotas $N(D)$',
                  fontsize=16, fontweight='bold')
        
        # Marca de agua del logo
        if smn_logo is not None :
            imagebox = OffsetImage(smn_logo, zoom=1.25, alpha=0.1)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='figure fraction', frameon=False, pad=0.0)
            fig.add_artist(ab)
            
        # Nota de copyright fija
        fig.text(0.75, 0.12, '© 2026 DPMAYSR-DNCIPS/SMN', ha='center', va='bottom', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        # Guardado de la figura
        output_filename = f'disdro_SMN-Dorrego_DSD_espectrograma_{fecha_obj.strftime("%Y%m%d")}.png'
        output_path = path_out.joinpath(output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Espectrograma DSD guardado en: {output_path}")
        
    finally :
        plt.close(fig)
        
# =========================================================================== #

def plot_resumen_historico(df, path_out, smn_logo=None):
    """
    Genera un gráfico resumen con la lluvia diaria acumulada, parámetros Z-R, 
    el porcentaje de registros (barras apiladas) y la disponibilidad de datos.
    El DataFrame debe tener: 'Fecha', 'Cantidad_Registros', 'Lluvia_mm', 
    'Dato_Disponible', 'ZR_param_a', 'ZR_param_b'.
    """
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    
    # Asegurar que la columna Fecha sea datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Calcular el porcentaje de completitud (asumiendo 1440 registros = 100%)
    if 'Cantidad_Registros' in df.columns:
        df['Porcentaje_Registros'] = (df['Cantidad_Registros'] / 1440.0) * 100.0
        # Limitar al 100% por si en algún caso hay más de 1440 registros
        df['Porcentaje_Registros'] = df['Porcentaje_Registros'].clip(upper=100)
    
    # Crear figura con CUATRO subplots apilados
    fig, (ax1, ax2, ax_pct, ax3) = plt.subplots(4, 1, 
                                   figsize=(12, 9), 
                                   gridspec_kw={'height_ratios': [4, 2.5, 1, 1], 'hspace': 0.08}, 
                                   sharex=True)
    
    try:
        # --- AX1: Serie temporal de Lluvia ---
        ax1.bar(df['Fecha'], df['Lluvia_mm'], color='dodgerblue', width=1.5)
        ax1.set_ylabel('Lluvia Diaria\n[mm]', fontsize=8, fontweight='bold')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax1.tick_params(axis='y', labelsize=6)

        ax1.set_title('Resumen Histórico de Observaciones', fontsize=14, pad=10)
 
        # --- AX2: Parámetros Z-R (Ejes Izquierdo y Derecho) ---
        if 'ZR_param_a' in df.columns and 'ZR_param_b' in df.columns:
            df_zr = df.dropna(subset=['ZR_param_a', 'ZR_param_b'])
            
            # Eje Y Izquierdo (Parámetro a)
            ax2.plot(df_zr['Fecha'], df_zr['ZR_param_a'], marker='o', markersize=4, 
                     color='purple', linestyle='', linewidth=1.5, alpha=0.8)
            ax2.set_ylabel('Parámetro a', fontsize=8, fontweight='bold', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)
            ax2.grid(True, axis='y', linestyle=':', alpha=0.5)
            
            # Eje Y Derecho (Parámetro b)
            ax2_b = ax2.twinx()
            ax2_b.plot(df_zr['Fecha'], df_zr['ZR_param_b'], marker='s', markersize=4, 
                       color='darkorange', linestyle='', linewidth=1.5, alpha=0.8)
            ax2_b.set_ylabel('Parámetro b', fontsize=8, fontweight='bold', color='darkorange')
            ax2_b.tick_params(axis='y', labelcolor='darkorange', labelsize=8)
        else:
            ax2.text(0.5, 0.5, 'Parámetros Z-R no disponibles', ha='center', va='center', color='gray')
            ax2.set_yticks([])

        # --- AX_PCT (NUEVO): Porcentaje de Registros (Stacked Bar) ---
        if 'Porcentaje_Registros' in df.columns:
            # Calculamos el porcentaje que falta para llegar al 100%
            porcentaje_faltante = 100.0 - df['Porcentaje_Registros']
            
            # Barra base (verde) - Porcentaje disponible
            ax_pct.bar(df['Fecha'], df['Porcentaje_Registros'], color='#2ca02c', width=1.0)
            
            # Barra superior (roja) - Porcentaje faltante, arranca donde termina la verde (bottom)
            ax_pct.bar(df['Fecha'], porcentaje_faltante, bottom=df['Porcentaje_Registros'], color='#d62728', width=1.0)
            
            ax_pct.set_ylabel('Datos (%)', fontsize=8, fontweight='bold')
            ax_pct.set_ylim(0, 100) # Fijamos el eje Y de 0 a 100 estricto
            ax_pct.set_yticks([0, 25, 50, 75, 100]) # Mostramos las marcas de 0, 50 y 100
            ax_pct.tick_params(axis='y', labelsize=8)
            ax_pct.grid(True, axis='y', linestyle=':', alpha=0.5)
        else:
            ax_pct.text(0.5, 0.5, 'Cantidad no disponible', ha='center', va='center', color='gray')
            ax_pct.set_yticks([])

        # --- AX3: Disponibilidad de Datos (Día válido general) ---
        colores = ['#2ca02c' if disp else '#d62728' for disp in df['Dato_Disponible']]
        
        ax3.bar(df['Fecha'], [1]*len(df), color=colores, width=1.0)
        ax3.set_yticks([]) 
        ax3.set_ylabel('Estado', fontsize=8, fontweight='bold')
        ax3.set_xlabel('Fecha', fontsize=8, fontweight='bold')
        
        # Formateo del eje X
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=15)) 
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90, ha='center', fontsize=8)
         
        # Marca de agua
        if smn_logo is not None:
            imagebox = OffsetImage(smn_logo, zoom=1.0, alpha=0.1)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='axes fraction', frameon=False, pad=0.0)
            ax1.add_artist(ab)
        
        # Guardar archivo
        output_filename = 'disdro_SMN-Dorrego_resumen_historico.png'
        output_path = path_out.joinpath(output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figura resumen guardada en: {output_path}")
        
    finally:
        plt.close(fig)

# =========================================================================== #
