!M3C 2018 Homework 3 Frederica Melbourne CID: 01068192
!This module contains four module variables and two subroutines;
!one of these routines must be developed for this assignment.
!Module variables--
! tr_b, tr_e, tr_g: the parameters b, e, and g in the tribe competition model
! numthreads: The number of threads that should be used in parallel regions within simulate2_omp
!
!Module routines---
! simulate2_f90: Simulate tribal competition over m trials. Return: all s matrices at final time
! and fc at nt+1 times averaged across the m trials.
! simulate2_omp: Same input/output functionality as simulate2.f90 but parallelized with OpenMP

module tribes
  use omp_lib
  implicit none
  integer :: numthreads
  real(kind=8) :: tr_b,tr_e,tr_g
contains

!Simulate m trials of Cooperator vs. Mercenary competition using the parameters, tr_b and tr_e.
!Input:
! n: defines n x n grid of villages
! nt: number of time steps
! m: number of trials
!Output:
! s: status matrix at final time step for all m trials
! fc_ave: fraction of cooperators at each time step (including initial condition)
! averaged across the m trials
subroutine simulate2_f90(n,nt,m,s,fc_ave)
  implicit none
  integer, intent(in) :: n,nt,m
  integer, intent(out), dimension(n,n,m) :: s
  real(kind=8), intent(out), dimension(nt+1) :: fc_ave
  integer :: i1,j1, t1, t2, clock_rate
  real(kind=8) :: n2inv
  integer, dimension(n,n,m) :: nb,nc
  integer, dimension(n+2,n+2,m) :: s2
  real(kind=8), dimension(n,n,m) :: f,p,a,pden,nbinv
  real(kind=8), dimension(n+2,n+2,m) :: f2,f2s2
  real(kind=8), allocatable, dimension(:,:,:) :: r !random numbers

  !call system_clock(t1)
  !---Problem setup----
  !Initialize arrays and problem parameters

  !initial condition
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0

  n2inv = 1.d0/dble(n*n)
  fc_ave(1) = sum(s)*(n2inv/m)

  s2 = 0
  f2 = 0.d0

  !Calculate number of neighbors for each point
  nb = 8
  nb(1,2:n-1,:) = 5
  nb(n,2:n-1,:) = 5
  nb(2:n-1,1,:) = 5
  nb(2:n-1,n,:) = 5
  nb(1,1,:) = 3
  nb(1,n,:) = 3
  nb(n,1,:) = 3
  nb(n,n,:) = 3

  nbinv = 1.d0/nb
  allocate(r(n,n,m))
  !---finished Problem setup---


  !----Time marching----
  do i1=1,nt

    call random_number(r) !Random numbers used to update s every time step

    !Set up coefficients for fitness calculation in matrix, a
    a = 1
    where(s==0)
      a=tr_b
    end where

    !create s2 by adding boundary of zeros to s
    s2(2:n+1,2:n+1,:) = s

    !Count number of C neighbors for each point
    nc = s2(1:n,1:n,:) + s2(1:n,2:n+1,:) + s2(1:n,3:n+2,:) + &
         s2(2:n+1,1:n,:)                  + s2(2:n+1,3:n+2,:) + &
         s2(3:n+2,1:n,:)   + s2(3:n+2,2:n+1,:)   + s2(3:n+2,3:n+2,:)

    !Calculate fitness matrix, f----
    f = nc*a
    where(s==0)
      f = f + (nb-nc)*tr_e
    end where
    f = f*nbinv
    !-----------

    !Calculate probability matrix, p----
    f2(2:n+1,2:n+1,:) = f
    f2s2 = f2*s2

    !Total fitness of cooperators in community
    p = f2s2(1:n,1:n,:) + f2s2(1:n,2:n+1,:) + f2s2(1:n,3:n+2,:) + &
           f2s2(2:n+1,1:n,:) + f2s2(2:n+1,2:n+1,:)  + f2s2(2:n+1,3:n+2,:) + &
          f2s2(3:n+2,1:n,:)   + f2s2(3:n+2,2:n+1,:)   + f2s2(3:n+2,3:n+2,:)

    !Total fitness of all members of community
    pden = f2(1:n,1:n,:) + f2(1:n,2:n+1,:) + f2(1:n,3:n+2,:) + &
           f2(2:n+1,1:n,:) + f2(2:n+1,2:n+1,:)  + f2(2:n+1,3:n+2,:) + &
          f2(3:n+2,1:n,:)   + f2(3:n+2,2:n+1,:)   + f2(3:n+2,3:n+2,:)


    p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g) !probability matrix
    !----------

    !Set new affiliations based on probability matrix and random numbers stored in R
    s = 0
    where (R<=p)
        s = 1
    end where

    fc_ave(i1+1) = sum(s)*(n2inv/m)

  end do

  !call system_clock(t2, clock_rate)

  !print *, dble(t2-t1)/dble(clock_rate)

end subroutine simulate2_f90

!Simulate m trials of Cooperator vs. Mercenary competition using the parameters, tr_b and tr_e.
!Same functionality as simulate2_f90, but parallelized with OpenMP
!Parallel regions should use numthreads threads.
!Input:
! n: defines n x n grid of villages
! nt: number of time steps
! m: number of trials
!Output:
! s: status matrix at final time step for all m trials
! fc_ave: fraction of cooperators at each time step (including initial condition)
! averaged across the m trials

!Comments:

!I chose to rewrite the function so that it looped over the m trials and then parallelize these.
!It was possible to do this as the m trials are independent of each other, therefore there is no
!data dependence. In contrast I could not parallelise the loop over the timesteps as each step requires
!data from the step before. I placed the loop over the trials outside the timestep loop so that the
!threads did not have to be put together and forked as often.

!I decided that parallelising over the m trials was a better option than unvectorising some of the parts
! within the timestep loop (e.g. probability/fitness matrix) and then parallelising these. Firstly this
! would mean creating parallel regions within a loop, which is inefficient. Secondly, by parallelising
! over the trials it meant most of the code could be parallelised, therefore leading to greater speedup.

!Finally, I used the reduction directive so that fc_ave could be calculated as all threads would need to
!change this variable.

subroutine simulate2_omp(n,nt,m,s,fc_ave)
  implicit none
  integer, intent(in) :: n,nt,m
  integer, intent(out), dimension(n,n,m) :: s
  real(kind=8), intent(out), dimension(nt+1) :: fc_ave
  integer :: i1,j1
  real(kind=8), allocatable, dimension(:,:,:) :: R !random numbers
  !Add further variables as needed
  integer :: j, clock_rate, t1, t2
  integer, dimension(n,n) :: nb,nc
  integer, dimension(n+2,n+2) :: s2
  real(kind=8), dimension(n,n) :: f,p,a,pden,nbinv
  real(kind=8), dimension(n+2,n+2) :: f2,f2s2
  real(kind=8) :: n2inv

  !call system_clock(t1)
  !initial condition and r allocation (does not need to be parallelized)
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0
  allocate(R(n,n,m))
  !------------------
  n2inv = 1.d0/dble(n*n)
  fc_ave(1) = sum(s)

  s2 = 0
  f2 = 0.d0

  !Calculate number of neighbors for each point
  nb = 8
  nb(1,2:n-1) = 5
  nb(n,2:n-1) = 5
  nb(2:n-1,1) = 5
  nb(2:n-1,n) = 5
  nb(1,1) = 3
  nb(1,n) = 3
  nb(n,1) = 3
  nb(n,n) = 3

  nbinv = 1.d0/nb

  !Add code here

  !$    call omp_set_num_threads(numthreads)

  !$OMP PARALLEL FIRSTPRIVATE(s2,f2) PRIVATE(i1,a,R,nc,f,f2s2,p,pden)
  !$OMP DO REDUCTION(+:fc_ave)
  do j = 1,m

    do i1 = 1,nt

      call random_number(R(:,:,j))

      a = 1
      where(s(:,:,j)==0)
        a=tr_b
      end where

      !create s2 by adding boundary of zeros to s
      s2(2:n+1,2:n+1) = s(:,:,j)

      !Count number of C neighbors for each point
      nc = s2(1:n,1:n) + s2(1:n,2:n+1) + s2(1:n,3:n+2) + &
           s2(2:n+1,1:n)                  + s2(2:n+1,3:n+2) + &
           s2(3:n+2,1:n)   + s2(3:n+2,2:n+1)   + s2(3:n+2,3:n+2)

      !Calculate fitness matrix, f----
      f = nc*a
      where(s(:,:,j)==0)
        f = f + (nb-nc)*tr_e
      end where
        f = f*nbinv

      !Calculate probability matrix, p----
      f2(2:n+1,2:n+1) = f
      f2s2 = f2*s2

      !Total fitness of cooperators in community
      p = f2s2(1:n,1:n) + f2s2(1:n,2:n+1) + f2s2(1:n,3:n+2) + &
          f2s2(2:n+1,1:n) + f2s2(2:n+1,2:n+1)  + f2s2(2:n+1,3:n+2) + &
          f2s2(3:n+2,1:n)   + f2s2(3:n+2,2:n+1)   + f2s2(3:n+2,3:n+2)

      !Total fitness of all members of community
      pden = f2(1:n,1:n) + f2(1:n,2:n+1) + f2(1:n,3:n+2) + &
             f2(2:n+1,1:n) + f2(2:n+1,2:n+1)  + f2(2:n+1,3:n+2) + &
             f2(3:n+2,1:n)   + f2(3:n+2,2:n+1)   + f2(3:n+2,3:n+2)

      p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g) !probability matrix

      s(:,:,j) = 0

      where (R(:,:,j)<=p)
          s(:,:,j) = 1
      end where

      fc_ave(i1+1) = fc_ave(i1+1) + sum(s(:,:,j))

    end do

  end do
  !$OMP END DO
  !$OMP END PARALLEL

  fc_ave = fc_ave*(n2inv/m)

  deallocate(R)

  !call system_clock(t2, clock_rate)

  !print *, dble(t2-t1)/dble(clock_rate)
end subroutine simulate2_omp


!-------------------------------------!
!- Function used to create animation -!
! (Rewritten version of simulate2_omp)!
! - Returns 3D array of one trial at -!
! -------- each time step ------------!
!-------------------------------------!

subroutine simulate22_f90(n,nt,evolve)
  implicit none
  integer, intent(in) :: n,nt
  integer, intent(out), dimension(n,n,nt+1) :: evolve
  integer, dimension(n,n) :: s
  integer :: i1,j1
  real(kind=8), allocatable, dimension(:,:) :: R !random numbers
  !Add further variables as needed
  integer :: i, j
  integer, dimension(n,n) :: nb, nc
  integer, dimension(n+2,n+2) :: s2
  real(kind=8), dimension(n,n) :: f,p,a,pden,nbinv
  real(kind=8), dimension(n+2,n+2) :: f2,f2s2
  real(kind=8) :: n2inv

  s=1
  j1 = (n+1)/2
  s(j1,j1) = 0
  allocate(R(n,n))

  evolve(:,:,1)=s
  !------------------
  n2inv = 1.d0/dble(n*n)

  s2 = 0
  f2 = 0.d0

  !Calculate number of neighbors for each point
  nb = 8
  nb(1,2:n-1) = 5
  nb(n,2:n-1) = 5
  nb(2:n-1,1) = 5
  nb(2:n-1,n) = 5
  nb(1,1) = 3
  nb(1,n) = 3
  nb(n,1) = 3
  nb(n,n) = 3

  nbinv = 1.d0/nb

  do i1 = 1,nt

    call random_number(R)

    a = 1
    where(s==0)
      a=tr_b
    end where

    !create s2 by adding boundary of zeros to s
    s2(2:n+1,2:n+1) = s

    !Count number of C neighbors for each point
    nc = s2(1:n,1:n) + s2(1:n,2:n+1) + s2(1:n,3:n+2) + &
         s2(2:n+1,1:n)                  + s2(2:n+1,3:n+2) + &
         s2(3:n+2,1:n)   + s2(3:n+2,2:n+1)   + s2(3:n+2,3:n+2)

    !Calculate fitness matrix, f----
    f = nc*a
    where(s==0)
      f = f + (nb-nc)*tr_e
    end where
      f = f*nbinv

    !Calculate probability matrix, p----
    f2(2:n+1,2:n+1) = f
    f2s2 = f2*s2

    !Total fitness of cooperators in community
    p = f2s2(1:n,1:n) + f2s2(1:n,2:n+1) + f2s2(1:n,3:n+2) + &
        f2s2(2:n+1,1:n) + f2s2(2:n+1,2:n+1)  + f2s2(2:n+1,3:n+2) + &
        f2s2(3:n+2,1:n)   + f2s2(3:n+2,2:n+1)   + f2s2(3:n+2,3:n+2)

    !Total fitness of all members of community
    pden = f2(1:n,1:n) + f2(1:n,2:n+1) + f2(1:n,3:n+2) + &
           f2(2:n+1,1:n) + f2(2:n+1,2:n+1)  + f2(2:n+1,3:n+2) + &
           f2(3:n+2,1:n)   + f2(3:n+2,2:n+1)   + f2(3:n+2,3:n+2)

    p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g) !probability matrix

    s = 0

    where (R<=p)
       s = 1
    end where

    print *, s

    evolve(:,:,i1+1) = s

  end do

  deallocate(R)

end subroutine simulate22_f90

end module tribes
