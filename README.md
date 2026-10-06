# HvA_Data_Science_Case_3

Informatie over vluchten vanuit Zürich in 2019 en 2020
Maak een kolom met vertragingen en bekijk de gegevens uit 06670.csv met betrekking tot het tijdstip, het vliegtuigtype, de landingsbaan, de bestemming, het actuele verkeersvolume en de weersgegevens. Kijk wanneer de vertragingen optreden en zoek naar gemeenschappelijke factoren. Wees in staat om de nauwkeurigheid en mogelijke fouten aan te wijzen. Gebruik de gegevens van 2019 en 2020 apart vanwege corona; zo kunt u nagaan of een situatie die in 2019 tot een vertraging leidde, ook in 2020 vertraging veroorzaakt.

Combineer de vluchtschema's en de luchthavenlijst van ICAO om een ​​kaart te maken van "Waar komen deze vliegtuigen aan of naartoe?". Het verschil is niet relevant, vertragingen hebben gevolgen voor zowel vertrekkende als aankomende vluchten.

Beperk de gegevens tot één type locatie in plaats van een volledige wereldkaart. Alleen binnen Europa, of alleen intercontinentale vluchten.

Maak handmatig een kolom voor vertragingen aan in de samenvoeging om het verschil tussen de geplande en daadwerkelijke aankomsttijden te zien, rekening houdend met vertragingen rond middernacht. Bijvoorbeeld: 23:55 en 00:05 is 10 minuten te laat, niet 23 uur te vroeg.

Kies een gebied voor uw kaart. 295 bestemmingen wereldwijd op één kaart resulteert in een wolk van stippen waar niemand wijs uit kan worden. Zoom in op iets dat een verhaal vertelt: alleen Europa, waar het grootste deel van het verkeer zich bevindt en u de dichtheid echt kunt zien, of alleen de intercontinentale vluchten, waar er weinig zijn maar elke vlucht telt. Die keuze is op zich al een ontwerpbeslissing en telt mee voor de beoordeling.

Vertraging is een kolom die u zelf aanmaakt. Kijk naar wat er rond middernacht gebeurt: een vlucht die gepland staat voor 23:55 en landt om 00:10 is 15 minuten te laat, niet 23 uur te vroeg.

Lees het bestand eenmaal in en cache het met @st.cache_data.
