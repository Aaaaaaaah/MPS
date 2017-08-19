program mps
    use tools
    use tensor_type

    integer      :: D = 8, t

    type(tensor) :: expH    ! Hamiltonian
    type(tensor) :: H       ! Hamiltonian
    type(tensor) :: II      ! Hamiltonian
    type(tensor) :: A       ! A
    type(tensor) :: B       ! B
    type(tensor) :: EAB     ! Environment between A and B
    type(tensor) :: EBA     ! Environment between B and A
    type(tensor) :: L       ! Left
    type(tensor) :: R       ! Right

    call init

    do t=1,1000
    call update
    end do

    do t=1,1000
    call update_L
    call update_R
    end do
    call Energy

    call save_data

contains

    subroutine update_L()
        type(tensor) :: tmp

        tmp = eye(EAB,D,D)
        call tmp%setName(1,'tmp.left')
        call tmp%setName(2,'tmp.right')

        L = contract(L,['L.1'],A,['A.left'])
        L = contract(L,['A.right'],tmp,['tmp.left'])
        L = contract(L,['tmp.right'],tmp,['tmp.left'])
        call L%setName('tmp.right','L.1')

        L = contract(L,['L.2','A.phy'],A,['A.left','A.phy'])
        L = contract(L,['A.right'],tmp,['tmp.left'])
        L = contract(L,['tmp.right'],tmp,['tmp.left'])
        call L%setName('tmp.right','L.2')

        tmp = eye(EBA,D,D)
        call tmp%setName(1,'tmp.left')
        call tmp%setName(2,'tmp.right')

        L = contract(L,['L.1'],B,['B.left'])
        L = contract(L,['B.right'],tmp,['tmp.left'])
        L = contract(L,['tmp.right'],tmp,['tmp.left'])
        call L%setName('tmp.right','L.1')

        L = contract(L,['L.2','B.phy'],B,['B.left','B.phy'])
        L = contract(L,['B.right'],tmp,['tmp.left'])
        L = contract(L,['tmp.right'],tmp,['tmp.left'])
        call L%setName('tmp.right','L.2')

        L = L/(L%dmaxmin('maxabs'))

    end subroutine update_L

    subroutine update_R()
        type(tensor) :: tmp

        tmp = eye(EAB,D,D)
        call tmp%setName(1,'tmp.left')
        call tmp%setName(2,'tmp.right')

        R = contract(R,['R.1'],B,['B.right'])
        R = contract(R,['B.left'],tmp,['tmp.left'])
        R = contract(R,['tmp.right'],tmp,['tmp.left'])
        call R%setName('tmp.right','R.1')

        R = contract(R,['R.2','B.phy'],B,['B.right','B.phy'])
        R = contract(R,['B.left'],tmp,['tmp.left'])
        R = contract(R,['tmp.right'],tmp,['tmp.left'])
        call R%setName('tmp.right','R.2')

        tmp = eye(EBA,D,D)
        call tmp%setName(1,'tmp.left')
        call tmp%setName(2,'tmp.right')

        R = contract(R,['R.1'],A,['A.right'])
        R = contract(R,['A.left'],tmp,['tmp.left'])
        R = contract(R,['tmp.right'],tmp,['tmp.left'])
        call R%setName('tmp.right','R.1')

        R = contract(R,['R.2','A.phy'],A,['A.right','A.phy'])
        R = contract(R,['A.left'],tmp,['tmp.left'])
        R = contract(R,['tmp.right'],tmp,['tmp.left'])
        call R%setName('tmp.right','R.2')

        R = R/(R%dmaxmin('maxabs'))

    end subroutine update_R

    subroutine Energy()
        type(Tensor) :: xHx, tmp

        tmp = eye(EAB,D,D)
        call tmp%setName(1,'tmp.left')
        call tmp%setName(2,'tmp.right')

        xHx = contract(L,['L.1'],A,['A.left'])
        call xHx%setName('A.phy','A.1')
        xHx = contract(xHx,['A.right'],tmp,['tmp.left'])
        xHx = contract(xHx,['tmp.right'],tmp,['tmp.left'])
            !call xHx%setName('A.right','tmp.right')
        xHx = contract(xHx,['tmp.right'],B,['B.left'])
        call xHx%setName('B.phy','B.1')
        xHx = contract(xHx,['B.right'],R,['R.1'])
        xHx = contract(xHx,['L.2'],A,['A.left'])
        call xHx%setName('A.phy','A.2')
        xHx = contract(xHx,['A.right'],tmp,['tmp.left'])
        xHx = contract(xHx,['tmp.right'],tmp,['tmp.left'])
            !call xHx%setName('A.right','tmp.right')
        xHx = contract(xHx,['tmp.right','R.2'],B,['B.left','B.right'])
        call xHx%setName('B.phy','B.2')

        !call xHx%write(6)
        print *, (xHx.ddot.H)/(xHx.ddot.II)

    end subroutine Energy

    subroutine save_data()
        open(1,file='L.dat')
        call L%write(1)
        close(1)

        open(1,file='R.dat')
        call R%write(1)
        close(1)

        open(1,file='A.dat')
        call A%write(1)
        close(1)

        open(1,file='B.dat')
        call B%write(1)
        close(1)

        open(1,file='EAB.dat')
        call EAB%write(1)
        close(1)

        open(1,file='EBA.dat')
        call EBA%write(1)
        close(1)
    end subroutine save_data

    subroutine init()
        integer i
        logical f

        ! Initialize Hamiltonian
        call expH%allocate([2,2,2,2],'real')
        expH = reshape([0.99999,0.,0.,0.,0.,1.00001,-0.00002,0.,0.,-0.00002,1.00001,0.,0.,0.,0.,0.99999],[2,2,2,2])
        call expH%setName(1,"expH.A1")
        call expH%setName(2,"expH.B1")
        call expH%setName(3,"expH.A2")
        call expH%setName(4,"expH.B2")

        call H%allocate([2,2,2,2],'real')
        H = reshape([0.25,0.,0.,0.,0.,-0.25,0.5,0.,0.,0.5,-0.25,0.,0.,0.,0.,0.25],[2,2,2,2])
        call H%setName(1,"A.1")
        call H%setName(2,"B.1")
        call H%setName(3,"A.2")
        call H%setName(4,"B.2")

        call II%allocate([2,2,2,2],'real')
        II = reshape([1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.],[2,2,2,2])
        call II%setName(1,"A.1")
        call II%setName(2,"B.1")
        call II%setName(3,"A.2")
        call II%setName(4,"B.2")

        call L%allocate([D,D],'real')
        inquire( file=trim('L.dat'), exist=f)
        if(f) then
            open(1,file='L.dat')
            call L%read(1)
            close(1)
        else
            call L%random()
            L = L - 0.5
            call L%setName(1,"L.1")
            call L%setName(2,"L.2")
        endif

        call R%allocate([D,D],'real')
        inquire( file=trim('R.dat'), exist=f)
        if(f) then
            open(1,file='R.dat')
            call R%read(1)
            close(1)
        else
            call R%random()
            R = R - 0.5
            call R%setName(1,"R.1")
            call R%setName(2,"R.2")
        endif

        ! Initialize A
        call A%allocate([D,D,2],'real')
        inquire( file=trim('A.dat'), exist=f)
        if(f) then
            open(1,file='A.dat')
            call A%read(1)
            close(1)
        else
            call A%random()
            A = A - 0.5
            call A%setName(1,"A.left")
            call A%setName(2,"A.right")
            call A%setName(3,"A.phy")
        endif

        ! Initialize B
        call B%allocate([D,D,2],'real')
        inquire( file=trim('B.dat'), exist=f)
        if(f) then
            open(1,file='B.dat')
            call B%read(1)
            close(1)
        else
            call B%random()
            B = B - 0.5
            call B%setName(1,"B.left")
            call B%setName(2,"B.right")
            call B%setName(3,"B.phy")
        endif

        ! Initialize EAB and EBAA
        call EAB%allocate([D],'real')
        inquire( file=trim('EAB.dat'), exist=f)
        if(f) then
            open(1,file='EAB.dat')
            call EAB%read(1)
            close(1)
        else
            do i=1,D
            call EAB%setValue([i],1.)
            end do
        endif

        call EBA%allocate([D],'real')
        inquire( file=trim('EBA.dat'), exist=f)
        if(f) then
            open(1,file='EBA.dat')
            call EBA%read(1)
            close(1)
        else
            do i=1,D
            call EBA%setValue([i],1.)
            end do
        endif

    end subroutine init

    subroutine update()
        type(tensor) :: tmp, acu, SVD(3)
        integer      :: i

        ! Contract
        tmp = eye(EBA,D,D)
        call tmp%setName(1,"tmp.left")
        call tmp%setName(2,"tmp.right")
        acu = contract(tmp,['tmp.right'],A,['A.left'])


        tmp = eye(EAB,D,D)
        call tmp%setName(1,"tmp.left")
        call tmp%setName(2,"tmp.right")
        acu = contract(acu,['A.right'],tmp,['tmp.left'])
        acu = contract(acu,['tmp.right'],tmp,['tmp.left'])


        acu = contract(acu,['tmp.right'],B,['B.left'])
        tmp = eye(EBA,D,D)
        call tmp%setName(1,"tmp.left")
        call tmp%setName(2,"tmp.right")
        acu = contract(acu,['B.right'],tmp,['tmp.left'])

        acu = contract(acu,['A.phy','B.phy'],expH,['expH.A1','expH.B1'])

        call acu%setName('tmp.left','A.left')
        call acu%setName('tmp.right','B.right')
        call acu%setName('expH.A2','A.phy')
        call acu%setName('expH.B2','B.phy')

        SVD=acu%SVDTensor('A','B',D)
        A = SVD(1)
        B = SVD(3)
        EAB = SVD(2) !!!!!!!!! sqrt

        do i = 1, D
        call EAB%setValue([i],sqrt(EAB%di([i])))
        end do

        call A%setName(3,'A.right')
        call B%setName(1,'B.left')

        tmp = eye(EBA,D,D)
        tmp = tmp%invTensor()
        call tmp%setName(1,'A.left')
        call tmp%setName(2,'B.right')

        A = contract(A,['A.left'],tmp,['B.right'])
        B = contract(B,['B.right'],tmp,['A.left'])

        A = A/(A%dmaxmin('maxabs'))
        B = B/(B%dmaxmin('maxabs'))
        EAB = EAB/(EAB%dmaxmin('maxabs'))
        EBA = EBA/(EBA%dmaxmin('maxabs'))


        !!!!!!!!!! another

        ! Contract
        tmp = eye(EAB,D,D)
        call tmp%setName(1,"tmp.left")
        call tmp%setName(2,"tmp.right")
        acu = contract(tmp,['tmp.right'],B,['B.left'])


        tmp = eye(EBA,D,D)
        call tmp%setName(1,"tmp.left")
        call tmp%setName(2,"tmp.right")
        acu = contract(acu,['B.right'],tmp,['tmp.left'])
        acu = contract(acu,['tmp.right'],tmp,['tmp.left'])


        acu = contract(acu,['tmp.right'],A,['A.left'])
        tmp = eye(EAB,D,D)
        call tmp%setName(1,"tmp.left")
        call tmp%setName(2,"tmp.right")
        acu = contract(acu,['A.right'],tmp,['tmp.left'])

        acu = contract(acu,['B.phy','A.phy'],expH,['expH.B1','expH.A1'])

        call acu%setName('tmp.left','B.left')
        call acu%setName('tmp.right','A.right')
        call acu%setName('expH.B2','B.phy')
        call acu%setName('expH.A2','A.phy')

        SVD=acu%SVDTensor('B','A',D)
        B = SVD(1)
        A = SVD(3)
        EBA = SVD(2) !!!!!!!!! sqrt

        do i = 1, D
        call EBA%setValue([i],sqrt(EBA%di([i])))
        end do

        call B%setName(3,'B.right')
        call A%setName(1,'A.left')

        tmp = eye(EAB,D,D)
        tmp = tmp%invTensor()
        call tmp%setName(1,'B.left')
        call tmp%setName(2,'A.right')

        B = contract(B,['B.left'],tmp,['A.right'])
        A = contract(A,['A.right'],tmp,['B.left'])

        A = A/(A%dmaxmin('maxabs'))
        B = B/(B%dmaxmin('maxabs'))
        EAB = EAB/(EAB%dmaxmin('maxabs'))
        EBA = EBA/(EBA%dmaxmin('maxabs'))

    end subroutine update


end program mps
