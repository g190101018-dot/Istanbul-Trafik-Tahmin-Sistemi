import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="İstanbul 4 Yıllık Trafik Tahmin Sistemi", layout="centered")

st.title("🚗 İstanbul Trafik Yoğunluğu Tahmin Paneli")
st.write("📊 4 Yıllık Büyük Veri (Big Data) ile Eğitilmiş Model")
st.markdown("---")

@st.cache_resource
def kaynaklari_yukle():
    # --- KURŞUN GEÇİRMEZ YOL AYARI ---
    # Bu dosyanın (arayuz.py) bulunduğu tam klasör yolunu alıyoruz
    su_anki_klasor = os.path.dirname(os.path.abspath(__file__))
    
    # Ana proje klasörüne çıkıp oradan src klasörüne net yol çiziyoruz
    model_yolu = os.path.join(su_anki_klasor, "..", "src", "trafik_modeli_4yil.pkl")
    referans_yolu = os.path.join(su_anki_klasor, "..", "src", "4yillik_referans.pkl")
    
    with open(model_yolu, 'rb') as dosya:
        model = pickle.load(dosya)
    with open(referans_yolu, 'rb') as f:
        paket = pickle.load(f)
        
    return model, paket['ref_tablo'], paket['min_arac'], paket['max_arac']

try:
    model, ref_tablo, min_arac, max_arac = kaynaklari_yukle()
    
    st.subheader("📅 Tahmin Ayarları")
    
    gunler = {
        "Pazartesi": 0, "Salı": 1, "Çarşamba": 2, "Perşembe": 3, 
        "Cuma": 4, "Cumartesi": 5, "Pazar": 6
    }
    secilen_gun = st.selectbox("Lütfen Bir Gün Seçiniz:", list(gunler.keys()))
    gun_kodu = gunler[secilen_gun]
    
    secilen_saat = st.select_slider(
        "Tahmin İstediğiniz Saati Belirleyin:",
        options=list(range(24)),
        format_func=lambda x: f"{x:02d}:00",
        value=12
    )
    
    st.markdown("---")
    
    if st.button("🚀 4 YILLIK VERİYE GÖRE TRAFİK DURUMUNU HESAPLA", use_container_width=True):
        bir_saat_once = (secilen_saat - 1) if secilen_saat > 0 else 23
        
        ref_veri = ref_tablo[(ref_tablo['gun_kod'] == gun_kodu) & (ref_tablo['saat'] == bir_saat_once)]
        
        if not ref_veri.empty:
            ortalama_gecmis_trafik = ref_veri['NUMBER_OF_VEHICLES'].values[0]
            
            hafta_sonu = 1 if gun_kodu >= 5 else 0
            X_input = pd.DataFrame([{
                'saat': secilen_saat,
                'gun_kod': gun_kodu,  
                'hafta_sonu': hafta_sonu,
                'bir_saat_once': ortalama_gecmis_trafik
            }])
            
            tahmin = model.predict(X_input)[0]
            
            yuzdelik_konum = (tahmin - min_arac) / (max_arac - min_arac)
            skor = int(np.clip(yuzdelik_konum * 10, 1, 10))
            
            if secilen_saat in [1, 2, 3, 4, 5]:
                skor = max(1, skor - 3)
            
            if secilen_saat in [17, 18, 19] and hafta_sonu == 0:
                skor = min(10, skor + 2)
            
            if skor <= 3:
                renk = "green"
                durum = "Yol Açık / Akıcı"
                emoji = "🟢"
            elif skor <= 7:
                renk = "orange"
                durum = "Orta Yoğunluk"
                emoji = "🟡"
            else:
                renk = "red"
                durum = "Kilit Trafik / Çok Yoğun"
                emoji = "🔴"
                
            st.markdown(f"<h2 style='text-align: center; color: {renk};'>{emoji} {durum}</h2>", unsafe_allow_html=True)
            
            bar_text = ""
            for i in range(1, 11):
                if i <= skor:
                    bar_text += "█"
                else:
                    bar_text += "░"
            
            st.markdown(f"<h1 style='text-align: center; letter-spacing: 5px; color: {renk};'>{bar_text}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Yoğunluk Seviyesi: {skor}/10</p>", unsafe_allow_html=True)
            
            if hafta_sonu == 0 and secilen_saat in [8, 9, 17, 18, 19]:
                st.warning("⚠️ Hafta içi mesai/iş saatidir. Yoğunluk değişkenlik gösterebilir.")
        else:
            st.error("Referans veri paketinde hata oluştu.")

except Exception as e:
    st.error(f"Sistem başlatılamadı: {e}")