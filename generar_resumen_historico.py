#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import linregress

# Rutas locales
path_input = Path('/home/lucho/SMN/disdro/output/daily/')
path_output = Path('/home/lucho/SMN/disdro/output/resumen_historico.csv')

def calcular_zr(df):
    """
    Filtra los datos diarios y calcula los parámetros a y b de la relación Z = a * R^b
    usando regresión lineal en espacio logarítmico (Z [dBZ] vs log10(R)).
    Retorna (a, b) si el ajuste es posible, o (None, None) si no hay datos suficientes.
    """
    if 'Reflectividad_dBZ' not in df.columns or 'Tasa_Lluvia_mm_h' not in df.columns:
        return None, None
        
    filtered_df = df[df['Reflectividad_dBZ'] != -99.9].copy()
    filtered_df = filtered_df[filtered_df['Tasa_Lluvia_mm_h'] > 0].copy()
    
    if len(filtered_df) < 2:
        return None, None
        
    filtered_df['log10_Tasa_Lluvia_mm_h'] = np.log10(filtered_df['Tasa_Lluvia_mm_h'])
    
    slope, intercept, r_value, p_value, std_err = linregress(
        filtered_df['log10_Tasa_Lluvia_mm_h'], 
        filtered_df['Reflectividad_dBZ']
    )
    
    # Z(dBZ) = 10 * log10(a) + 10 * b * log10(R)
    # intercept = 10 * log10(a)  => a = 10^(intercept/10)
    # slope = 10 * b             => b = slope / 10
    
    param_a = np.power(10, intercept/10)
    param_b = slope / 10
    
    return round(param_a, 2), round(param_b, 2)

def generar_resumen():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando generación de resumen histórico...")
    
    # 1. Buscar todos los archivos para determinar la fecha del primer registro
    archivos = list(path_input.glob('salida_disdrometro_*.csv'))
    
    if not archivos:
        print("Error: No se encontraron archivos diarios en el directorio.")
        sys.exit(1)
        
    fechas_disponibles = []
    for arch in archivos:
        try:
            # Extrae YYYYMMDD de 'salida_disdrometro_YYYYMMDD.csv'
            fecha_str = arch.stem.split('_')[-1]
            fecha_obj = datetime.strptime(fecha_str, '%Y%m%d').date()
            fechas_disponibles.append(fecha_obj)
        except ValueError:
            continue
            
    if not fechas_disponibles:
        print("Error: No se pudieron parsear las fechas de los archivos.")
        sys.exit(1)
        
    # 2. Definir rango continuo de fechas (desde el archivo más viejo hasta hoy)
    fecha_inicio = min(fechas_disponibles)
    fecha_fin = datetime.now().date()
    rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    
    datos_resumen = []
    
    # 3. Recorrer cada día del calendario
    for fecha in rango_fechas:
        fecha_str = fecha.strftime('%Y%m%d')
        archivo = path_input.joinpath(f'salida_disdrometro_{fecha_str}.csv')
        
        lluvia_total = 0.0
        disponible = False
        param_a, param_b = None, None
        
        # 4. Validar si existe y extraer el dato
        if archivo.exists():
            try:
                df = pd.read_csv(archivo)
                if not df.empty and 'Lluvia_Acumulada_Total_mm' in df.columns:
                    # Extraemos el último valor de lluvia acumulada
                    lluvia_total = df['Lluvia_Acumulada_Total_mm'].iloc[-1]
                    disponible = True
                    
                    # Calcular Z-R para el día
                    param_a, param_b = calcular_zr(df)
            except Exception as e:
                print(f"Error leyendo {archivo.name}: {e}")
        
        datos_resumen.append({
                              'Fecha': fecha.strftime('%Y-%m-%d'),
                              'Lluvia_mm': round(lluvia_total, 2),
                              'ZR_param_a': param_a,
                              'ZR_param_b': param_b,
                              'Dato_Disponible': disponible  
                             })
        
    # 5. Generar y exportar el DataFrame
    df_resumen = pd.DataFrame(datos_resumen)
    df_resumen.to_csv(path_output, index=False)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resumen guardado exitosamente en: {path_output}")

if __name__ == '__main__':
    generar_resumen()