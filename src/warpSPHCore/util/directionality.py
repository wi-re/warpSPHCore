import warp as wp

@wp.func
def checkDirectionality_j(
    queryKind: wp.int32, opInt: wp.int32
):
    if opInt == 0: # No Ghost
        return queryKind != 2
    elif opInt == 9: # All to all
        return queryKind != 2
    elif opInt == 1: # fluid to fluid
        return queryKind == 0
    elif opInt == 2: # fluid to boundary
        return queryKind == 0
    elif opInt == 3: # boundary to fluid
        return queryKind == 1
    elif opInt == 4: # boundary to boundary
        return queryKind == 1
    elif opInt == 5: # fluid to ghost
        return queryKind == 0
    elif opInt == 6: # ghost to fluid
        return queryKind == 2
    elif opInt == 7: # boundary to ghost
        return queryKind == 1
    elif opInt == 8: # ghost to boundary
        return queryKind == 2
    elif opInt == 10: # all to ghost
        return queryKind != 2
    elif opInt == 11: # all to fluid
        return queryKind != 2
    elif opInt == 12: # all to boundary
        return queryKind != 2
    elif opInt == 13: # fluid to all
        return queryKind == 0
    elif opInt == 14: # boundary to all
        return queryKind == 1
    else:
        return False

@wp.func
def checkDirectionality_i(
    referenceKind: wp.int32, opInt: wp.int32
):
    if opInt == 0: # No Ghost
        return referenceKind != 2
    elif opInt == 9: # All to all
        return True
    elif opInt == 1: # fluid to fluid
        return referenceKind == 0
    elif opInt == 2: # fluid to boundary
        return referenceKind == 1
    elif opInt == 3: # boundary to fluid
        return referenceKind == 0
    elif opInt == 4: # boundary to boundary
        return referenceKind == 1
    elif opInt == 5: # fluid to ghost
        return referenceKind == 2
    elif opInt == 6: # ghost to fluid
        return referenceKind == 0
    elif opInt == 7: # boundary to ghost
        return referenceKind == 2
    elif opInt == 8: # ghost to boundary
        return referenceKind == 1
    elif opInt == 10: # all to ghost
        return referenceKind == 2
    elif opInt == 11: # all to fluid
        return referenceKind == 0
    elif opInt == 12: # all to boundary
        return referenceKind == 1
    elif opInt == 13: # fluid to all
        return referenceKind != 2
    elif opInt == 14: # boundary to all
        return referenceKind != 2
    else:
        return False
    
@wp.func
def checkDirectionality_Func(
    queryKind: wp.int32, referenceKind: wp.int32, opInt: wp.int32
):
    return checkDirectionality_i(queryKind, opInt) and checkDirectionality_j(referenceKind, opInt)