# Transformasi dari Teks Menjadi Representasi Embedding Kata
Catatan ini dibuat untuk memberikan gambaran singkat tentang bagaimana mengubah (kumpulan) dokumen teks menjadi representasi embedding kata (numerik) untuk keperluan lebih lanjut. Lebih lanjut mengenai apa itu embedding kata ada sedikit penjelasan pada catatan pada link berikut: [Bagaimanakah Embedding Layer berfungsi?
](https://github.com/rizalespe/Catatan-Deep-Learning/blob/master/Embedding-Layer.md#bagaimanakah-embedding-layer-berfungsi).  Berikut adalah beberapa urutan proses yang dilakukan:
1. Pra proses teks standar (_cleaning_, _tokenizing_, normalisasi, dll)
2. Menyusun dokumen teks menjadi _list_ dokumen
3. Membuat daftar pasangan kata beserta indeks berupa bilangan bulat numerik
4. Transformasi setiap kata pada _list_ dokumen (hasil dari langkah 2) menjadi bilangan numerik berdasarkan daftar kata yang telah disusun pada langkah 3
5. Instansiasi objek Embedding layer. Pada catatan ini, saya menggunakan framework deep learning Pytorch. Berikut dokumentasi resmi [Pytorch Embedding](https://pytorch.org/docs/stable/nn.html#embedding)
6. Dalam kumpulan dokumen (corpus), setiap dokumen memiliki jumlah kata yang berbeda-beda. Untuk menyeragamkan jumlah kata tersebut dapat kita lakukan padding. Lebih jelasnya pada pada dokumentasi resmi [Pytorch pad_sequence](https://pytorch.org/docs/stable/nn.html#pad-sequence)
7. Langkah no. 6 menghasilkan tensor yang kemudian akan menjadi input pada embedding layer yang telah diinstansiasi pada proses no. 5