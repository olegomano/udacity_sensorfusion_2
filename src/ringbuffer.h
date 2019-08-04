#ifndef _RING_BUFFER_H_
#define _RING_BUFFER_H_
#include <cinttypes>
#include <iterator>

template<typename T>
class RingBuffer{
public:
    class iterator{
        public:
            typedef T value_type;
            typedef std::ptrdiff_t          difference_type;
            typedef T*                      pointer;
            typedef T&                      reference;
            typedef std::input_iterator_tag iterator_category;

            explicit iterator(RingBuffer<T>& buffer,uint32_t ptr) :
                m_owner(buffer),
                m_ptr(ptr)
                {}

            T& operator*() const { return m_owner.get(m_ptr); }
            T* operator->() const { return ((T*)&m_owner.get(m_ptr)); }
            bool operator==(const RingBuffer<T>::iterator& other) const { return m_ptr == other.m_ptr; }
            bool operator!=(const RingBuffer<T>::iterator& other) const { return !(*this == other); }
        
            iterator operator+(int incremet){
                return iterator(m_owner,m_ptr + incremet);
            }

            iterator operator-(int decrement){
                return iterator(m_owner,m_ptr - decrement);
            }


            iterator& operator++(){
                ++m_ptr;
                return *this;
            }
        private:
            RingBuffer<T>& m_owner;
            uint32_t       m_ptr;

    };
    

    RingBuffer(int size) : 
    m_size(size),
    m_buffer(new T[size])
    {

    }

    void push_back(const T& item){
        m_buffer[m_ptr % m_size] = item;
        ++m_ptr;
    }

    const T& get(const uint32_t indx){
        return m_buffer[indx % m_size];
    }

    iterator begin(){
        return iterator(*this,m_ptr);
    }

    iterator end(){
        return iterator(*this,m_ptr + m_size);
    }

    uint32_t size(){
        return m_ptr;
    }

private:
    const int m_size;
    T*        m_buffer;
    uint32_t  m_ptr = 0;
};

#endif