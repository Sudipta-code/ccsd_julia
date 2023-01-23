using Distributed
using TensorOperations,SharedArrays
using Base.Threads
using PyCall
using Einsum,LinearAlgebra,LoopVectorization
using DependencyWalker, LibSSH2_jll
using HDF5

println("number of threads ", nthreads())


Lovdf = h5open("Lov_df.h5", "r") do file_Lov
    read(file_Lov, "/Lov")
end

Lvvdf = h5open("Lvv_df.h5", "r") do file_Lvv
    read(file_Lvv, "/Lvv")
end

t1 = h5open("t1.h5", "r") do file_t1
    read(file_t1, "/t1")
end

tau = h5open("tau.h5", "r") do file_tau
    read(file_tau, "/tau")
end

nocc,nvir = size(t1)
t2new = zeros(nocc,nocc,nvir,nvir)
temp = zeros(nvir,nvir)


function build_temp(t1,nocc,nvir,Lovdf,Lvvdf,tau)
    tempf = zeros(nvir,nvir)
    Lov_T120 = permutedims(Lovdf,(2,3,1))
    local t2new_temp = Array{Float64,4}(undef,(nocc,nocc,nvir,nvir))
    local temp = Array{Float64,2}(undef,(nvir,nvir))
    temps = [zeros(nvir,nvir) for i = 1:Threads.nthreads()]
    t2new_temps = [zeros(nocc,nocc,nvir,nvir) for i = 1:Threads.nthreads()]
    Threads.@threads for i in 1:nocc
        @inbounds begin
            id = Threads.threadid()
            temp = temps[id]
            t2new_temp =  t2new_temps[id]
            for j in 1:nocc
                @views tau_ij_ = tau[i,j,:,:]
                @tensoropt (k=>x, t=>100x, d=>100x, c=>100x, a=>100x, b=>100x) begin
                    temp[a,b] = (-t1[k,b]*Lov_T120[k,d,t]*tau_ij_[c,d]*Lvvdf[t,a,c])
                    temp[a,b] -= t1[k,a]*Lov_T120[k,c,t]*tau_ij_[c,d]*Lvvdf[t,b,d]
                    temp[a,b] += Lvvdf[t,a,c]*tau_ij_[c,d]*Lvvdf[t,b,d]
                end
                t2new_temp[i,j,:,:] += temp
            end
        end
    end #sync
    temp_final= sum(temps)
    t2new_final = sum(t2new_temps)
    println("temp from multithreaded code ")
    display(temp_final)
    display(t2new_final)
    temp_final
end



function build_temp_serial(t1,nocc,nvir,Lovdf,Lvvdf,tau)
    Lov_T120 = permutedims(Lovdf,(2,3,1))
    t2new_temp = Array{Float64,4}(undef,(nocc,nocc,nvir,nvir))
    temp = Array{Float64,2}(undef,(nvir,nvir))
    for i in 1:nocc
        @inbounds begin
            for j in 1:nocc
                @views tau_ij_ = tau[i,j,:,:]
                @tensoropt (k=>x, t=>100x, d=>100x, c=>100x, a=>100x, b=>100x) begin
                    temp[a,b] = (-t1[k,b]*Lov_T120[k,d,t]*tau_ij_[c,d]*Lvvdf[t,a,c])
                    temp[a,b] -= t1[k,a]*Lov_T120[k,c,t]*tau_ij_[c,d]*Lvvdf[t,b,d]
                    temp[a,b] += Lvvdf[t,a,c]*tau_ij_[c,d]*Lvvdf[t,b,d]
                end
                t2new[i,j,:,:] += temp
            end
        end
    end #sync
    println("temp from serial code ")
    display(temp)
    display(t2new)
    temp
end

temp_final = build_temp(t1,nocc,nvir,Lovdf,Lvvdf,tau)
temp = build_temp_serial(t1,nocc,nvir,Lovdf,Lvvdf,tau)


