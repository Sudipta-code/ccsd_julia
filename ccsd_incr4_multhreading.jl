using Distributed
using TensorOperations,SharedArrays
using Base.Threads
using PyCall
using IterTools
using Einsum,LinearAlgebra,LoopVectorization
using DependencyWalker, LibSSH2_jll
using HDF5
using ChunkSplitters
using Atomix: @atomic, @atomicswap, @atomicreplace

println("number of threads ", nthreads())

function int_copy_store(t1::AbstractArray{T,2}, eris::PyObject) where T<: AbstractFloat
    nocc, nvir = size(t1)
    ovvv = Array{Float64,4}(undef,(nocc,nvir,nvir,nvir))

    ovov = similar(eris.ovov, Float64, nocc,nvir,nocc,nvir)
    oovv = similar(eris.oovv, Float64, nocc,nocc,nvir,nvir)
    ovvv = similar(eris.ovvv, Float64, nocc,nvir,nvir,nvir)
    ovoo = similar(eris.ovoo, Float64, nocc,nvir,nocc,nocc)
    oooo = similar(eris.oooo, Float64, nocc,nocc,nocc,nocc)


    ovov .= eris.ovov
    oovv .= eris.oovv
    ovvv .= eris.ovvv
    ovoo .= eris.ovoo
    oooo .= eris.oooo
    return ovov, oovv, ovvv, ovoo, oooo
end

function integral_df(t1::AbstractArray{T,2}, eris::PyObject) where T<: AbstractFloat
    nocc, nvir = size(t1)
    naux = eris.naux

    Loo_df = similar(eris.Loo)
    Lov_df = similar(eris.Lov)
    Lvv_df = similar(eris.Lvv)

    Loo_df .= eris.Loo
    Lov_df .= eris.Lov
    Lvv_df .= eris.Lvv

    return Loo_df, Lov_df, Lvv_df
end


function fock_slice(nocc::Int64,nvir::Int64, eris::PyObject) #where T<: AbstractFloat
    nbasis = nocc + nvir
    fock = eris.fock[:,:]
    foo = fock[1:nocc, 1:nocc]
    fov = fock[1:nocc, nocc+1:nbasis]
    fvo = fock[nocc+1:nbasis, 1:nocc]
    fvv = fock[nocc+1:nbasis, nocc+1:nbasis]
    return fock, foo, fov, fvo, fvv
end

function e_pairs(nocc::Int64)
    pair_ls = Tuple{Int,Int}[]
    @fastmath @inbounds @simd for i in 1:nocc
        @fastmath @inbounds @simd for j in 1:i
            @fastmath @inbounds push!(pair_ls, (i, j))
        end
    end
    @fastmath @inbounds return pair_ls
end

function mul_e_pairs(nocc::Int64)
    pair_ls1 = Tuple{Int,Int}[]
    pair_ls2 = Tuple{Int,Int}[]
    for i in range(1, nocc)
        for j in range(1, i)
            if i == j
                push!(pair_ls1, (i, j))
            else
                push!(pair_ls2, (i, j))
            end
        end
    end
    return pair_ls1, pair_ls2
end

function build_temp_multhrd(t1,nocc,nvir,Lov_df,Lvv_df,tau)
    nocc, nvir = size(t1)
    tempf = zeros(nvir,nvir)
    Lov_T120 = permutedims(Lov_df,(2,3,1))
    local t2new_temp = Array{Float64,4}(undef,(nocc,nocc,nvir,nvir))
    local temp = Array{Float64,2}(undef,(nvir,nvir))
    temps = [zeros(nvir,nvir) for i = 1:Threads.nthreads()]
    #BLAS_THREADS = BLAS.get_num_threads()
    #BLAS.set_num_threads(1)
    Threads.@threads for i in 1:nocc
        @inbounds begin
            id = Threads.threadid()
            temp = temps[id]
            for j in 1:nocc
                @views tau_ij_ = tau[i,j,:,:]
                @tensor begin
                    temp[a,b] += (-t1[k,b]*Lov_T120[k,d,t]*tau_ij_[c,d]*Lvv_df[t,a,c])
                    temp[a,b] -= t1[k,a]*Lov_T120[k,c,t]*tau_ij_[c,d]*Lvv_df[t,b,d]
                    temp[a,b] += Lvv_df[t,a,c]*tau_ij_[c,d]*Lvv_df[t,b,d]
                end
            end
        end
    end #sync
    #end #spawn
    #temp_t = reduce(+, temps)
    temp_final= sum(temps)
    #BLAS.set_num_threads(BLAS_THREADS)
    println("temp from multithreaded code ")
    display(temp_final)
    return temp_final
end

#=
function build_temp(t1,nocc,nvir,Lov_df,Lvv_df,tau)
    Lov_T120 = permutedims(Lov_df,(2,3,1))
    nchunks = Threads.nthreads()
    temps = [zeros(nvir,nvir ) for i = 1:nchunks]
    Threads.@threads for (i_range, ichunk) in chunks(1:nocc, nchunks)
        @inbounds begin
            temp = temps[ichunk]
            for i in i_range
                for j in 1:i
                @views tau_ij_ = tau[i,j,:,:]
                @tensor begin
                    temp[a,b] = (-t1[k,b]*Lov_T120[k,d,t]*tau_ij_[c,d]*Lvv_df[t,a,c])
                    temp[a,b] -= t1[k,a]*Lov_T120[k,c,t]*tau_ij_[c,d]*Lvv_df[t,b,d]
                    temp[a,b] += Lvv_df[t,a,c]*tau_ij_[c,d]*Lvv_df[t,b,d]
                end
                end
            end
        end
    end
    final_result = sum(temps)
    #=
    final_result = zeros(nvir, nvir)
    temp_list = [(i, temps[i]) for i in 1:Threads.nthreads()]
    temp_list = sort(temp_list, by=x->x[1])
    final_result = reduce(+, [x[2] for x in temp_list])
    =#
    display(final_result)
    return final_result
end
=#


function pair_check(nocc::Int64, Lov_df::Array{Float64,3}, Lvv_df::Array{Float64,3})#,t1::Array{Float64,2},eris::PyObject)
    for i in 1:nocc
        for j in 1:i
            @fastmath @inbounds Lov_T120 = permutedims(Lov_df,(2,3,1))
            @fastmath tau_ij_ = @views tau[i,j,:,:]
            @tensor ttmp1[k,t,c] = Lov_T120[k,d,t]*tau_ij_[c,d]
            @tensor ttmp2[k,a] = ttmp1[k,t,c]*Lvv_df[t,a,c]
            @tensor temp[a,b] = (-t1[k,b]*ttmp2[k,a])
            @tensor ttmp3[k,t,d] = Lov_T120[k,c,t]*tau_ij_[c,d]
            @tensor ttmp4[k,b] = ttmp3[k,t,d]*Lvv_df[t,b,d]
            @tensor temp[a,b] -= t1[k,a]*ttmp4[k,b]
            @tensor ttmp5[t,a,d] = Lvv_df[t,a,c]*tau_ij_[c,d]
            @tensor temp[a,b] += ttmp5[t,a,d]*Lvv_df[t,b,d]
        end
    end
    return temp
end

function cc_energy(t1,t2,ovov,fov)
    nocc, nvir = size(t1)
    nbasis = nocc+nvir
    e = 0.0
    tau = zeros(Float64,nocc,nocc,nvir,nvir)
    @tensor tau[i,j,a,b] = (t1[i,a]*t1[j,b])  #qudratic term
    tau += t2
    @tensor e = 2*(fov[i,a]*t1[i,a])
    @tensor e += 2*(tau[i,j,a,b]*ovov[i,a,j,b])
    @tensor e -= tau[i,j,a,b]*ovov[i,b,j,a]
    return real(e)
end


function cc_iter(t1,t2,eris, cc_input)
    nocc, nvir = size(t1)
    ECCSD = 0.0
    ovov, oovv,ovvv,ovoo, oooo= int_copy_store(t1,eris)
    Loo_df, Lov_df, Lvv_df = integral_df(t1,eris)
    fock,foo,fov,fvo,fvv = fock_slice(nocc,nvir, eris)

    @inbounds for j in 1:66
        println("Iteration ", j)
        OLDCC = ECCSD
        t1, t2 = update_cc_amps(t1,t2,eris,ovov,oovv,ovvv,ovoo,oooo,fock,foo, fov, fvo, fvv,Lov_df,Lvv_df,Loo_df)
        ECCSD = cc_energy(t1,t2,ovov,fov)
        DECC = abs(ECCSD - OLDCC)
        println(" DECC  = ", DECC)
        println(" ECCSD = ", ECCSD)
        println("   ")
        println("   ")
        convergence = 1.0e-12

        if DECC < convergence
            println("TOTAL ITERATIONS: ", j)
            break
        end
    end
    ECCSD = cc_energy(t1,t2,ovov,fov)
    println("Final CCSD correlation energy  ", ECCSD)
    return t1, t2
end

function update_cc_amps(t1::AbstractArray{T,2},t2::AbstractArray{T,4},eris,ovov::AbstractArray{T,4},
    oovv::AbstractArray{T,4},ovvv::AbstractArray{T,4},ovoo::AbstractArray{T,4},
    oooo::AbstractArray{T,4},fock::AbstractArray{T,2},foo::AbstractArray{T,2},
    fov::AbstractArray{T,2}, fvo::AbstractArray{T,2},fvv::AbstractArray{T,2},
    Lov_df::AbstractArray{T,3},Lvv_df::AbstractArray{T,3},Loo_df::AbstractArray{T,3}) where T<:AbstractFloat

    level_shift = 0.00
    @fastmath @inbounds nocc, nvir = size(t1)
    @fastmath @inbounds nbasis = nocc + nvir
    @fastmath @inbounds naux = eris.naux
    Loo_df, Lov_df, Lvv_df = integral_df(t1,eris)

    @fastmath @inbounds mo_e_o = eris.mo_energy[1:nocc]
    @fastmath @inbounds mo_e_v = eris.mo_energy[nocc+1:nbasis] .+ level_shift

    Foo = cc_Foo(t1, t2,ovov,foo)
    Fvv = cc_Fvv(t1,t2,ovov,fvv)
    Fov = cc_Fov(t1,t2,ovov,fov)


    @fastmath @inbounds Foo[diagind(Foo)] -= mo_e_o
    @fastmath @inbounds Fvv[diagind(Fvv)] -= mo_e_v

    #------------- T1 equation -----------------#

    ksht = Array{Float64,2}(undef, nvir, nvir)
    t1new = Array{Float64,2}(undef, nocc, nvir)
    ttnew = Array{Float64,3}(undef, naux, nocc, nvir)
    tautemp = Array{Float64,4}(undef, nocc, nocc, nvir, nvir)


    fov_T = permutedims(fov)
    println(" Time for 1st block of T1 and T2 using @time macro ")
    @time begin
        @tensor begin
            ksht[c,a] = fov_T[c,k]*t1[k,a]
            t1new[i,a] = -2*(t1[i,c]*ksht[c,a])
            t1new[i,a] += Fvv[a,c]*t1[i,c]
            t1new[i,a] -= Foo[k,i]*t1[k,a]
        end
        t1new += conj(fov)

        @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>100x, b=>100x, c=>100x, d=>100x) begin
            t1new[i,a] += 2*(Fov[k,c]*t2[k,i,c,a])
            t1new[i,a] -= Fov[k,c]*t2[i,k,c,a]
            t1new[i,a] += 2*(ovov[k,c,i,a]*t1[k,c])  #keep it on .............
            t1new[i,a] -= oovv[k,i,a,c]*t1[k,c]
            t1new[i,a] += Fov[k,c]*t1[i,c]*t1[k,a]    #qudratic term
        end

        if eris.incore < 4 && eris.df
            @tensoropt (i=>x, k=>x, c=>100x, d=>100x) begin
                tautemp[i,k,c,d]=t1[k,d]*t1[i,c]
            end
            tautemp += t2
            @tensoropt (i=>x, k=>x, m=>100x, a=>100x, c=>100x, d=>100x) begin
                ttnew[m,i,c] = Lov_df[m,k,d]*tautemp[i,k,c,d]
                t1new[i,a] += ttnew[m,i,c]*Lvv_df[m,a,c]
            end
            delete!(ttnew)
            @tensoropt (i=>x, k=>x, m=>100x, a=>100x, c=>100x, d=>100x) begin
                ttnew[m,i,d] = Lov_df[m,k,c]*tautemp[i,k,c,d]
                t1new[i,a] -= ttnew[m,i,d]*Lvv_df[m,a,d]
            end
            delete!(ttnew)
            delete!(tautemp)

        else
            @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>100x, b=>100x, c=>100x, d=>100x) begin
                t1new[i,a] += 2*(ovvv[k,d,a,c]*t2[i,k,c,d])
                t1new[i,a] -= ovvv[k,c,a,d]*t2[i,k,c,d]
                t1new[i,a] += 2*(ovvv[k,d,a,c]*t1[k,d]*t1[i,c])  #qudratic term
                t1new[i,a] -= ovvv[k,c,a,d]*t1[k,d]*t1[i,c]      #qudratic term
            end
        end

        @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>100x, b=>100x, c=>100x, d=>100x) begin
            t1new[i,a] -= 2*(ovoo[l,c,k,i]*t2[k,l,a,c])
            t1new[i,a] += ovoo[k,c,l,i]*t2[k,l,a,c]
        end

        kstz = zeros(Float64,nocc,nocc)
        @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>10x, b=>10x, c=>10x, d=>10x) begin
            kstz[i,k] = ovoo[l,c,k,i]*t1[l,c]
            t1new[i,a] -= 2*(kstz[i,k]*t1[k,a])
            kstz[i,k] = ovoo[k,c,l,i]*t1[l,c]
            t1new[i,a] += kstz[i,k]*t1[k,a]
        end



        # ----------T2 Equation ------------ #
        tmp2_prime = Array{Float64,3}(undef, naux, nocc, nvir)
        tmp2 = zeros(Float64,nvir,nvir,nocc,nvir)
        tmp = zeros(Float64,nocc,nocc,nvir,nvir)
        ovvv_T = zeros(Float64,nvir,nvir,nocc,nvir)

        t1_minus = -t1

        if eris.incore < 4 && eris.df
            tmp4[k,i,j,b] = oovv[k,i,b,c]*(-t1[j,c])
            tmp[i,j,a,b]= tmp4[k,i,j,b]*t1[k,a]
            delete!(tmp4)
            tmp2_prime[m,j,b] = Lvv_df[m,c,b]*t1[m,j,b]
            tmp += transpose(Lov_df)[a,i,m]*tmp2_prime[m,j,b]
            delete!(tmp2_prime)

        else
            @tensor tmp2[a,b,i,c] = oovv[k,i,b,c]*(-t1[k,a])
            tmp2 += permutedims(ovvv,(2,4,1,3))
            @tensor tmp[i,j,a,b] = tmp2[a,b,i,c]*t1[j,c]

        end
        t2new = tmp + permutedims(tmp,(2,1,4,3))

        tmp2 = zeros(Float64,nvir,nocc,nocc,nocc)
        @tensor tmp2[a,k,i,j] = ovov[k,c,i,a]*t1[j,c]
        tmp2 += permutedims(ovoo,(2,4,1,3))
        tmp = zeros(nocc,nocc,nvir,nvir)
        @tensor tmp[i,j,a,b] = tmp2[a,k,i,j]*t1[k,b]
        t2new -= tmp + permutedims(tmp,(2,1,4,3))
        t2new += permutedims(ovov,(1,3,2,4))
    end

    Loo = Loioi(t1,t2,ovoo,ovov,fov,foo)
    Lvv = Lvirvir(t1,t2,ovvv,ovov,fvv,fov)
    Loo[diagind(Foo)] -= mo_e_o

    Lvv[diagind(Fvv)] -= mo_e_v

    Woooo = cc_Woooo(t1,t2,ovoo,ovov,oooo)
    Wvoov = cc_Wvoov(t1,t2,ovvv,ovov,ovoo)
    Wvovo = cc_Wvovo(t1,t2,ovvv,ovoo,oovv,ovov)

    tau = zeros(Float64,nocc,nocc,nvir,nvir)
    tmp = zeros(Float64,nocc,nocc,nvir,nvir)

    @tensor tau[i,j,a,b] = t1[i,a]*t1[j,b]
    tau += t2

    @tensor t2new[i,j,a,b] += Woooo[k,l,i,j]*tau[k,l,a,b]



    if eris.incore < 5 && eris.df
        temp = build_temp_multhrd(t1,nocc,nvir,Lov_df,Lvv_df,tau)
        exit()
        @inbounds @simd for i in 1:nocc
            @inbounds @simd for j in 1:i
                t2new[i,j,:,:] += temp
                if i!=j
                    t2new[j,i,:,:] += transpose(temp)
                end
            end
        end
    else
        println(" time taken for cc_Wvvvv function @time macro")
        @time Wvvvv = cc_Wvvvv(t1,t2,ovvv,vvvv)
        @tensor t2new[i,j,a,b] += Wvvvv[a,b,c,d]*tau[i,j,c,d]
    end

    @tensor tmp[i,j,a,b] = Lvv[a,c]*t2[i,j,c,b]
    t2new += (tmp + permutedims(tmp,(2,1,4,3)))
    tmp = nothing

    tmp = zeros(Float64,nocc,nocc,nvir,nvir)
    @tensor tmp[i,j,a,b] = Loo[k,i]*t2[k,j,a,b]
    t2new -= (tmp +permutedims(tmp,(2,1,4,3)))

    tmp = nothing

    tmp = zeros(Float64,nocc,nocc,nvir,nvir)
    @tensor tmp[i,j,a,b] = 2*(Wvoov[a,k,i,c]*t2[k,j,c,b])
    @tensor tmp[i,j,a,b] -= Wvovo[a,k,c,i]*t2[k,j,c,b]
    t2new += (tmp + permutedims(tmp,(2,1,4,3)))

    tmp = nothing


    tmp = zeros(Float64,nocc,nocc,nvir,nvir)
    @tensor tmp[i,j,a,b] = (Wvoov[a,k,i,c]*t2[k,j,b,c])
    t2new -= (tmp + permutedims(tmp,(2,1,4,3)))
    tmp = zeros(Float64,nocc,nocc,nvir,nvir)
    @tensor tmp[i,j,a,b] = Wvovo[b,k,c,i]*t2[k,j,a,c]
    t2new -= (tmp + permutedims(tmp,(2,1,4,3)))
    tmp = nothing


    py"""
    import numpy as np
    def denomi(a,b,t1new,t2new):
        eia = a[:, None] - b
        eijab = eia[:, None, :, None] + eia[None, :, None, :]
        t1new1 = t1new/eia
        t2new1 = t2new/eijab
        return t1new1, t2new1
    """
    t1new1,t2new1 = py"denomi"(mo_e_o,mo_e_v,t1new,t2new)


    return t1new1, t2new1
end

# ----- rintermediates ------------------- #
function cc_Foo(t1, t2,ovov,foo)
    nocc, nvir = size(t1)
    Fki = zeros(Float64,nocc,nocc)

    tautemp = zeros(Float64,nocc,nocc,nvir,nvir)
    @tensor tautemp[i,l,c,d] = t1[i,c]*t1[l,d]    #qudratic term
    tautemp += t2
    @tensor Fki[k,i] = 2*(ovov[k,c,l,d]*tautemp[i,l,c,d])
    @tensor Fki[k,i] -= ovov[k,d,l,c]*tautemp[i,l,c,d]
    Fki += foo
    return Fki
end

function cc_Fvv(t1,t2,ovov,fvv)
    nocc, nvir = size(t1)
    nbasis = nocc + nvir
    tautemp = zeros(Float64,nocc,nocc,nvir,nvir)
    Fac = zeros(Float64,nvir,nvir)
    @tensor tautemp[k,l,a,d] = t1[k,a]*t1[l,d]    #qudratic term
    tautemp += t2
    @tensor Fac[a,c] = (ovov[k,d,l,c]*tautemp[k,l,a,d])
    @tensor Fac[a,c] -= 2*(ovov[k,c,l,d]*tautemp[k,l,a,d])
    Fac += copy(fvv)
    return Fac
end

function cc_Fov(t1,t2,ovov,fov)
    nocc, nvir = size(t1)
    nbasis = nocc + nvir
    Fkc = zeros(Float64,nocc,nvir)
    @tensor Fkc[k,c] = 2*(ovov[k,c,l,d]*t1[l,d])
    @tensor Fkc[k,c] -= ovov[k,d,l,c]*t1[l,d]
    Fkc += fov
    return Fkc
end

function Loioi(t1,t2,ovoo,ovov,fov,foo)
    nocc, nvir = size(t1)
    nbasis = nocc + nvir
    Lki = cc_Foo(t1, t2,ovov,foo)
    @tensor Lki[k,i] += fov[k,c]*t1[i,c]
    @tensor Lki[k,i] += 2*(ovoo[l,c,k,i]*t1[l,c])
    @tensor Lki[k,i] -= ovoo[k,c,l,i]*t1[l,c]
    return Lki
end

function Lvirvir(t1,t2,ovvv,ovov,fvv,fov)
    nocc, nvir = size(t1)
    nbasis = nocc + nvir
    Lac = cc_Fvv(t1,t2,ovov,fvv)
    @tensor Lac[a,c] -= fov[k,c]*t1[k,a]
    @tensor Lac[a,c] += 2*(ovvv[k,d,a,c]*t1[k,d])
    @tensor Lac[a,c] -= ovvv[k,c,a,d]*t1[k,d]
    return Lac
end

function cc_Woooo(t1,t2,ovoo,ovov,oooo)
    nocc,nvir = size(t1)
    Wklij = zeros(Float64,nocc,nocc,nocc,nocc)
    oooo_T = zeros(Float64,nocc,nocc,nocc,nocc)
    @tensor Wklij[k,l,i,j] = ovoo[l,c,k,i]*t1[j,c]
    @tensor Wklij[k,l,i,j] += ovoo[k,c,l,j]*t1[i,c]
    @tensor Wklij[k,l,i,j] += ovov[k,c,l,d]*t2[i,j,c,d]
    @tensor Wklij[k,l,i,j] += ovov[k,c,l,d]*t1[i,c]*t1[j,d]  #quadratic
    Wklij += permutedims(oooo,(1,3,2,4))

    return Wklij
end

function cc_Wvvvv(t1::AbstractArray{T,2}, t2::AbstractArray{T,4}, ovvv::AbstractArray{T,4}, vvvv::AbstractArray{T,4}) where T<:AbstractFloat
    nocc, nvir = size(t1)
    Wabcd = zeros(T, nvir, nvir, nvir, nvir)
    @tensoropt (a=>x, b=>y, c=>z, d=>w, k=>u) begin
        Wabcd[a,b,c,d] = ovvv[k,d,a,c]*(-t1[k,b])
        Wabcd[a,b,c,d] -= ovvv[k,c,b,d]*t1[k,a]
    end
    Wabcd += permutedims(vvvv,(1,3,2,4))
    return Wabcd
end


function cc_Wvoov(t1,t2,ovvv,ovov,ovoo)
    nocc, nvir = size(t1)
    Wakic = zeros(Float64,nvir,nocc,nocc,nvir)
    @tensor Wakic[a,k,i,c]= ovvv[k,c,a,d]*t1[i,d]
    ovov_T = zeros(Float64,nvir,nocc,nocc,nvir)
    Wakic += permutedims(ovov,(4,1,3,2))
    @tensor Wakic[a,k,i,c] -= ovoo[k,c,l,i]*t1[l,a]
    @tensor Wakic[a,k,i,c] -= 0.5*(ovov[l,d,k,c]*t2[i,l,d,a])
    @tensor Wakic[a,k,i,c] -= 0.5*(ovov[l,c,k,d]*t2[i,l,a,d])
    @tensor Wakic[a,k,i,c] -= ovov[l,d,k,c]*t1[i,d]*t1[l,a] #quadratic
    @tensor Wakic[a,k,i,c] += ovov[l,d,k,c]*t2[i,l,a,d]
    return Wakic
end

function cc_Wvovo(t1,t2,ovvv,ovoo,oovv,ovov)

    nocc, nvir = size(t1)
    Wakci = zeros(Float64,nvir,nocc,nvir,nocc)
    oovv_T = zeros(Float64,nvir,nocc,nvir,nocc)
    @tensor Wakci[a,k,c,i] = ovvv[k,d,a,c]*t1[i,d]
    @tensor Wakci[a,k,c,i]  -= ovoo[l,c,k,i]*t1[l,a]
    Wakci += permutedims(oovv,(3,1,4,2))
    @tensor Wakci[a,k,c,i] -= 0.5*(ovov[l,c,k,d]*t2[i,l,d,a])
    @tensor Wakci[a,k,c,i] -= ovov[l,c,k,d]*t1[i,d]*t1[l,a] #qudratic term
    return Wakci
end