#=
ccsd_incore4b:
- Julia version:
- Author: sudipta
- Date: 2023-01-03
=#
#=
ccsd:
- Julia version:
- Author: sudipta
- Date: 2022-11-10
=#
using PyCall
using Einsum,LinearAlgebra,LoopVectorization
using TensorOperations,SharedArrays
using DependencyWalker, LibSSH2_jll
using Base.Threads
using Base.Threads
using Distributed
#using PythonCall
np = PyCall.pyimport("numpy")
time = PyCall.pyimport("time")
#denom = PyCall.pyimport("/home/sudipta/bagh_install/bagh_1/bagh/bagh_code/ccsd/denom.py")

println("number of threads ", nthreads())
#=
function int_copy_store(t1,eris)
    nocc,nvir = size(t1)
    ovov_tmp = eris.ovov
    oovv_tmp = eris.oovv
    ovvv_tmp = eris.ovvv
    ovoo_tmp = eris.ovoo
    oooo_tmp = eris.oooo
    vvvv_tmp = eris.vvvv



    ovov = Array{Float64,4}(undef,(nocc,nvir,nocc,nvir))
    oovv = Array{Float64,4}(undef,(nocc,nocc,nvir,nvir))
    ovvv = Array{Float64,4}(undef,(nocc,nvir,nvir,nvir))
    ovoo = Array{Float64,4}(undef,(nocc,nvir,nocc,nocc))
    oooo = Array{Float64,4}(undef,(nocc,nocc,nocc,nocc))
    vvvv = Array{Float64,4}(undef,(nvir,nvir,nvir,nvir))


    #=
     @inbounds for a in 1:nvir
        vvvv[a,:,:,:]= vvvv_tmp[a,:,:,:]
     end

    @inbounds for i in 1:nocc
        oovv[i,:,:,:] =  oovv_tmp[i,:,:,:]
    end


    @inbounds for i in 1:nocc
        ovov[i,:,:,:] =  ovov_tmp[i,:,:,:]
    end

    @inbounds for i in 1:nocc
        ovoo[i,:,:,:] =  ovoo_tmp[i,:,:,:]
    end

    @inbounds for i in 1:nocc
        oooo[i,:,:,:] =  oooo_tmp[i,:,:,:]
    end


    @inbounds for i in 1:nocc
        ovvv[i,:,:,:] =  ovvv_tmp[i,:,:,:]
    end
    =#
    vvvv[1:nvir,:,:,:] .= vvvv_tmp[1:nvir,:,:,:]
    oovv[1:nocc,:,:,:] .= oovv_tmp[1:nocc,:,:,:]
    ovov[1:nocc,:,:,:] .= ovov_tmp[1:nocc,:,:,:]
    ovoo[1:nocc,:,:,:] .= ovoo_tmp[1:nocc,:,:,:]
    oooo[1:nocc,:,:,:] .= oooo_tmp[1:nocc,:,:,:]
    ovvv[1:nocc,:,:,:] .= ovvv_tmp[1:nocc,:,:,:]
    return ovov, oovv,ovvv,ovoo, oooo,vvvv
end
=#


function int_copy_store(t1::AbstractArray{T,2}, eris::PyObject) where T<: AbstractFloat
    nocc, nvir = size(t1)
    ovvv = Array{Float64,4}(undef,(nocc,nvir,nvir,nvir))

    ovov = similar(eris.ovov, Float64, nocc,nvir,nocc,nvir)
    oovv = similar(eris.oovv, Float64, nocc,nocc,nvir,nvir)
    ovvv = similar(eris.ovvv, Float64, nocc,nvir,nvir,nvir)
    #ovvv_tmp = eris.ovvv
    ovoo = similar(eris.ovoo, Float64, nocc,nvir,nocc,nocc)
    oooo = similar(eris.oooo, Float64, nocc,nocc,nocc,nocc)
    #vvvv = similar(eris.vvvv)

    ovov .= eris.ovov
    oovv .= eris.oovv
    ovvv .= eris.ovvv
    #ovvv[1:nocc,:,:,:] .= ovvv_tmp[1:nocc,:,:,:]
    ovoo .= eris.ovoo
    oooo .= eris.oooo
    #vvvv .= eris.vvvv

    return ovov, oovv, ovvv, ovoo, oooo#, vvvv
end

function integral_df(t1,eris)
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

global pair_check

function build_temp(t1,nocc,nvir,Lov_df,Lvv_df,tau,t2new)
        @fastmath @inbounds temp  = Array{Float64,2}(undef,(nvir,nvir))

        @fastmath @inbounds Lov_T120 = permutedims(Lov_df,(2,3,1))

        #for (i,j) in pair_ls
        @inbounds @simd for i in 1:nocc
            @inbounds @simd for j in 1:i

                @fastmath tau_ij_ = @views tau[i,j,:,:]

                 @inbounds begin
                    @tensoropt (k=>x, t=>100x, d=>100x, c=>100x, a=>100x, b=>100x) begin
                        temp[a,b] = (-t1[k,b]*Lov_T120[k,d,t]*tau_ij_[c,d]*Lvv_df[t,a,c])
                        temp[a,b] -= t1[k,a]*Lov_T120[k,c,t]*tau_ij_[c,d]*Lvv_df[t,b,d]
                        temp[a,b] += Lvv_df[t,a,c]*tau_ij_[c,d]*Lvv_df[t,b,d]
                    end #@tensoropt

                    t2new[i,j,:,:] += temp
                    #=
                    if i!=j
                        t2new[j,i,:,:] += transpose(temp)
                    end
                    =#
                 end  #@inbounds
            end #@fastmath @inbounds @simd
        end #@fastmath @inbounds @simd
        #display(t2new)
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
    println(" Time to store integrals in julia array ")
    @time ovov, oovv,ovvv,ovoo, oooo= int_copy_store(t1,eris)
    println(" Time to store df_integrals in julia array ")
    @time Loo_df, Lov_df, Lvv_df = integral_df(t1,eris)
    #@time int_copy_store(t1,eris)
    fock,foo,fov,fvo,fvv = fock_slice(nocc,nvir, eris)

    #t2new = Array{Float64,4}(undef, (nocc, nocc, nvir, nvir))
    #temp_incr4  = Array{Float64,2}(undef,(nvir,nvir))

    @inbounds for j in 1:60
        println("Iteration ", j)
        OLDCC = ECCSD

        start1 = time.time()
        #--- timing of update_cc_amps in julia ----#
        println(" Time for update_cc_amps function using @time macro ")
        @time t1, t2 = update_cc_amps(t1,t2,eris,ovov,oovv,ovvv,ovoo,oooo,fock,foo, fov, fvo, fvv,Lov_df,Lvv_df,Loo_df)
        #t1 = np.zeros_like(t1)
        println(" Time cc_energy function using @time macro ")
        @time ECCSD = cc_energy(t1,t2,ovov,fov)
        end1 = time.time()
        println("time of iteration ", (end1-start1))
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

    level_shift = 0.00  # A shift on virtual orbital energies to stablize the CCSD iteration
    @fastmath @inbounds nocc, nvir = size(t1)
    @fastmath @inbounds nbasis = nocc + nvir
    @fastmath @inbounds naux = eris.naux



    @fastmath @inbounds mo_e_o = eris.mo_energy[1:nocc]
    @fastmath @inbounds mo_e_v = eris.mo_energy[nocc+1:nbasis] .+ level_shift
    println(" time taken for cc_Foo function @time macro")
    @time Foo = cc_Foo(t1, t2,ovov,foo)

    println(" time taken for cc_Fvv function @time macro")
    @time Fvv = cc_Fvv(t1,t2,ovov,fvv)

    println(" time taken for cc_Fov function @time macro")
    @time Fov = cc_Fov(t1,t2,ovov,fov)


    @fastmath @inbounds Foo[diagind(Foo)] -= mo_e_o


    @fastmath @inbounds Fvv[diagind(Fvv)] -= mo_e_v



    #fock = fock[1:nocc,nocc+1:nbasis]

    #------------- T1 equation -----------------#

    ksht = Array{Float64,2}(undef, nvir, nvir)
    t1new = Array{Float64,2}(undef, nocc, nvir)
    ttnew = Array{Float64,3}(undef, naux, nocc, nvir)
    tautemp = Array{Float64,4}(undef, nocc, nocc, nvir, nvir)


    fov_T = permutedims(fov)
    println(" Time for 1st block of T1 and T2 using @time macro ")
    @time begin
        @tensoropt (i=>x,k=>x,a=>100x,c=>100x) begin
            ksht[c,a] = fov_T[c,k]*t1[k,a]
            #println("fov_T ", fov_T)
            t1new[i,a] = -2*(t1[i,c]*ksht[c,a])
            t1new[i,a] += Fvv[a,c]*t1[i,c]
            t1new[i,a] -= Foo[k,i]*t1[k,a]
        end
        #println("t1new ")
        #display(t1new)
        #t1new += conj(fock[1:nocc,nocc+1:nbasis])
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


    println(" time taken for LoiLoi function @time macro")
    @time Loo = Loioi(t1,t2,ovoo,ovov,fov,foo)
    println(" time taken for Lvirvir function @time macro")
    @time Lvv = Lvirvir(t1,t2,ovvv,ovov,fvv,fov)
    Loo[diagind(Foo)] -= mo_e_o

    Lvv[diagind(Fvv)] -= mo_e_v


    println(" time taken for cc_Woooo function @time macro")
    @time Woooo = cc_Woooo(t1,t2,ovoo,ovov,oooo)
    println(" time taken for cc_Wvoov function @time macro")
    @time Wvoov = cc_Wvoov(t1,t2,ovvv,ovov,ovoo)
    println(" time taken for cc_Wvovo function @time macro")
    @time Wvovo = cc_Wvovo(t1,t2,ovvv,ovoo,oovv,ovov)

    tau = zeros(Float64,nocc,nocc,nvir,nvir)
    #t2new = zeros(nocc,nocc,nvir,nvir)
    tmp = zeros(Float64,nocc,nocc,nvir,nvir)

    @tensor tau[i,j,a,b] = t1[i,a]*t1[j,b]
    tau += t2

    @tensor t2new[i,j,a,b] += Woooo[k,l,i,j]*tau[k,l,a,b]

    if eris.incore < 5 && eris.df
        if eris.NumProc > 1
            global pair_check
            PL1, PL2 = mul_e_pairs(nocc)
            temp = pair_check(dummy)

            t21_temp = pmap(pair_check, PL1)
            t21_temp2 = pmap(pair_check, PL2)

            m = 1
            for (i, j) in PL1
                t2new[i, j, :, :] += t21_temp[m]
                m += 1
            end

            m = 1
            for (i, j) in PL2
                t2new[i, j, :, :] += t21_temp2[m]
                t2new[j, i, :, :] += t21_temp2[m]
                m += 1
            end

        else
            pair_ls = e_pairs(nocc)
            #println(pair_ls)
            @fastmath @inbounds ttmp1 = Array{Float64,3}(undef,(nocc,naux,nvir))
            @fastmath @inbounds ttmp2 = Array{Float64,2}(undef,(nocc,nvir))
            @fastmath @inbounds ttmp3 = Array{Float64,3}(undef,(nocc,naux,nvir))
            @fastmath @inbounds ttmp4 = Array{Float64,2}(undef,(nocc,nvir))
            @fastmath @inbounds ttmp5 = Array{Float64,3}(undef, (naux,nvir,nvir))
            @fastmath @inbounds temp  = Array{Float64,2}(undef,(nvir,nvir))


            @fastmath @inbounds Lov_T120 = permutedims(Lov_df,(2,3,1))

            #for (i,j) in pair_ls
            @inbounds @simd for i in 1:nocc
                @inbounds @simd for j in 1:nocc

                    println("i , j = ",i ," , ", j)

                    @fastmath tau_ij_ = @views tau[i,j,:,:]

                     @inbounds begin
                        @tensoropt (k=>x, t=>100x, d=>100x, c=>100x, a=>100x, b=>100x) begin
                            #=
                            ttmp1[k,t,c] = Lov_T120[k,d,t]*tau_ij_[c,d]
                            ttmp2[k,a]  = ttmp1[k,t,c]*Lvv_df[t,a,c]
                            temp[a,b] = (-t1[k,b]*ttmp2[k,a])

                            ttmp3[k,t,d] = Lov_T120[k,c,t]*tau_ij_[c,d]
                            ttmp4[k,b]  = ttmp3[k,t,d]*Lvv_df[t,b,d]
                            temp[a,b] -= t1[k,a]*ttmp4[k,b]

                            ttmp5[t,a,d] = Lvv_df[t,a,c]*tau_ij_[c,d]
                            temp[a,b] += ttmp5[t,a,d]*Lvv_df[t,b,d]
                            =#
                            temp[a,b] = (-t1[k,b]*Lov_T120[k,d,t]*tau_ij_[c,d]*Lvv_df[t,a,c])
                            temp[a,b] -= t1[k,a]*Lov_T120[k,c,t]*tau_ij_[c,d]*Lvv_df[t,b,d]
                            temp[a,b] += Lvv_df[t,a,c]*tau_ij_[c,d]*Lvv_df[t,b,d]
                        end #@tensoropt

                        t2new[i,j,:,:] += temp
                        if i!=j
                            t2new[j,i,:,:] += transpose(temp)
                        end

                    end  #@inbounds

                end #@fastmath @inbounds @simd
            end #@fastmath @inbounds @simd
            display(temp)
            exit()
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

    #println("t1new1 ", display(t1new1))

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
    #ovov = eris.ovov
    #ovov,oovv,ovvv,ovoo,oooo,vvvv = int_copy_store(t1,eris)
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
    #eris_ovvv = np.asarray(eris.get_ovvv())
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
    #eris_ovvv = np.asarray(eris.get_ovvv())
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
    #eris_ovvv = np.asarray(eris.get_ovvv())
    @tensor Wakci[a,k,c,i] = ovvv[k,d,a,c]*t1[i,d]
    @tensor Wakci[a,k,c,i]  -= ovoo[l,c,k,i]*t1[l,a]
    Wakci += permutedims(oovv,(3,1,4,2))
    @tensor Wakci[a,k,c,i] -= 0.5*(ovov[l,c,k,d]*t2[i,l,d,a])
    @tensor Wakci[a,k,c,i] -= ovov[l,c,k,d]*t1[i,d]*t1[l,a] #qudratic term
    return Wakci
end
#=
function Wooov(t1,t2,ovov,ovoo)
    nocc,nvir = size(t1)
    #=
    ovov = eris.ovov
    ovoo = eris.ovoo
    =#
    #ovov,oovv,ovvv,ovoo,oooo,vvvv = int_copy_store(t1,eris)
    @tensor Wklid[k,l,i,d] = t1[i,c]*ovov[k,c,l,d]
    #Wklid += np.asarray(eris.ovoo)#.transpose(2, 0, 3, 1)
    #Wklid += ovoo
    for i in 1:nocc
        for j in 1:nvir
            for k in 1:nocc
                for l in 1:nocc
                ovoo_T[k,i,l,j] = ovoo[i,j,k,l]
                end
            end
        end
    end
    Wklid += ovoo_T
    return Wklid
end

function Wvovv(t1,t2,ovov,ovvv)
    nocc, nvir = size(t1)
    #=
    ovov = eris.ovov
    ovvv = eris.ovvv
    =#
    #ovov,oovv,ovvv,ovoo,oooo,vvvv = int_copy_store(t1,eris)
    @tensor Walcd[a,l,c,d] = (-t1[k,a])*ovov[k,c,l,d]
    #Walcd_asarray = np.asarray(eris.get_ovvv())
    for i in 1:nocc
        for j in 1:nvir
            for k in 1:nvir
                for l in 1:nvir
                    ovvv_T[k,i,l,j] = ovvv[i,j,k,l]
                end
            end
        end
    end
    Walcd += ovvv_T
    return Walcd
end
function denomin(t1,t2,eris)
    nocc,nvir = size(t1)
    nbasis = nocc+nvir
    fock = eris.fock
    foo = eris.fock[1:nocc, 1:nocc]
    fvv = eris.fock[nocc+1:nbasis,nocc+1:nbasis]
    #x = zeros(nocc,nvir)
    D = []
    for i in 1:nocc
        for a in 1:nvir
            x=fock[i,i]- fock[a,a]
            push!(D,x)
        end
    end
    return D
end

#--timing of functions-------
=#




