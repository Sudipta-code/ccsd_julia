import numpy as np

#t1 = np.random.rand(5,2)
#nocc,nvir = t1.shape
#Lov_df = np.random.rand(60,5,2)
#Lvv_df = np.random.rand(60,2,2)
#tau = np.random.rand(5,5,2,2)

from julia import Main
Main.include("./multithreading_tensor.jl")
#temp_multithread = Main.build_temp(t1,nocc,nvir,Lov_df,Lvv_df,tau)
#temp = Main.build_temp_serial(t1,nocc,nvir,Lov_df,Lvv_df,tau)

#print(np.array_equal(temp_multithread, temp))
#print(np.array(temp_multithread) == np.array(temp))