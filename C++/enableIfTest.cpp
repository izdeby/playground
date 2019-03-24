#include <iostream>
#include <type_traits>

class Scalar {
public:
  Scalar() : Scalar(int(0)) {}

  Scalar (int vv) : tag(Tag::HAS_int) {
    std::cout << "int constructor" << std::endl;
    v.intProp = vv;
  }

  Scalar (float vv) : tag(Tag::HAS_float) {
    std::cout << "float constructor" << std::endl;
    v.floatProp = vv;
  }

  template <typename T,
            typename std::enable_if<std::is_same<T, bool>::value, bool>::type = nullptr>
  Scalar (T vv) : tag(Tag::HAS_bool) {
    std::cout << "bool constructor" << std::endl;
    v.boolProp = vv;
  }

  void GetMyType() {
    if (this->tag == Tag::HAS_int) {
      std::cout << "Im int! Value: " << v.intProp << std::endl;
    }

    if (this->tag == Tag::HAS_float) {
      std::cout << "Im float! Value: " << v.floatProp << std::endl;
    }

    if (this->tag == Tag::HAS_bool) {
      std::cout << "Im bool! Value: " << v.boolProp << std::endl;
    }
  }

private:

  enum class Tag { HAS_int, HAS_bool, HAS_float };
  Tag tag;

  union {
    float floatProp;
    int intProp;
    bool boolProp;
  } v;
};

int main() {
  auto intScalar = new Scalar(1);
  intScalar->GetMyType();

  auto floatScalar = new Scalar(2.2f);
  floatScalar->GetMyType();

  auto boolScalar = new Scalar(true);
  boolScalar->GetMyType();

  boolScalar = new Scalar(false);
  boolScalar->GetMyType();
  return 0;
}
