#include <iostream>
#include <functional>
#include <GroupAllocator/groupallocator>
#include <gtest/gtest.h>

TEST(GroupAllocator_test, TestForACrash) {
    groupallocator::Context ctx;

    char *s;
    char *t;

    groupallocator::allocate(&s, 40, ctx);

    for (unsigned long long i = 0; i < 39; i++) {
        s[i] = 'a';
    }
    s[39] = '\0';

    groupallocator::allocate(&t, 40, ctx);

    for (unsigned long long i = 0; i < 39; i++) {
        t[i] = 'b';
    }
    t[39] = '\0';

    groupallocator::free(t);

    groupallocator::freeall();
}

TEST(GroupAllocator_test, TooSmallAllocateInMPA) {
    ASSERT_TRUE(sizeof(int) + alignof(int *) > sizeof(int *));

    groupallocator::GroupAllocator g(0, sizeof(int));
    for (int i = 0; i < 10000; i++) {
        int *j;
        g.allocate(&j, sizeof(int), false);
        ASSERT_FALSE(j == nullptr);
        *j = 1;
        ASSERT_TRUE(g.pagesAllocated() == i + 1) << g.pagesAllocated() << " == " << i + 1;
    }
    g.freeall();
}

TEST(GroupAllocator_test, RepeatAllocateInIPA) {
    ASSERT_FALSE(sizeof(int) + alignof(int *) > 2 * sizeof(int *));
    groupallocator::GroupAllocator g(0, 2 * sizeof(int *));
    for (int i = 0; i < 10000; i++) {
        int *j;
        g.allocate(&j, sizeof(int), false);
        ASSERT_FALSE(j == nullptr);
        *j = 1;
    }
    g.freeall();
}

TEST(GroupAllocator_test, RepeatAllocatePtrToPtrInIPA) {
    groupallocator::GroupAllocator g(0, 128);
    for (int i = 0; i < 10000; i++) {
        int **j;
        g.allocate(&j, sizeof(int *), false);
        ASSERT_FALSE(j == nullptr);
        g.allocate(&j[0], sizeof(int), false);
    }
    g.freeall();
}

TEST(GroupAllocator_test, IPADoesntAllocateSamePtr) {
    groupallocator::GroupAllocator g(0, 128);
    for (int i = 0; i < 10000; i++) {
        int *j, *k;
        g.allocate(&j, sizeof(int), false);
        ASSERT_FALSE(j == nullptr);
        g.allocate(&k, sizeof(int), false);
        ASSERT_FALSE(k == nullptr);
        ASSERT_FALSE(j == k);
    }
    g.freeall();
}

TEST(GroupAllocator_test, MoveToGPU) {
    groupallocator::GroupAllocator g(0, 128);

    int *j, *k;
    g.allocate(&j, sizeof(int), false);
    g.moveToDevice(0, 0x0);

    g.allocate(&k, sizeof(char) * 4096, false);
    g.moveToDevice(0, 0x0);

    g.freeall();
}
