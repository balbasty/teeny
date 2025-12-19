
#ifndef MINITEN__DISP_DECL
#define MINITEN__DISP_DECL
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)

/** @brief  A structure that encapsulates static visualization tools. is
 *          redeclared as a function than calls the method internally.
 */
template <class Type>
struct Display;


/** @brief Print a new line */
MINIDEF(H,D,I) void newline();

/** @brief Print the representation of a type */
template <typename T>
MINIDEF(H,D,I) void disp();

/** @brief Print the representation of an instance */
template <typename T>
MINIDEF(H,D,I) void disp(const T value);

/** @brief Return the representation of a type */
template <typename T>
MINIDEF(H,D,I,CX) const char * srepr();

/** @brief Return the representation of an instance */
template <typename T>
MINIDEF(H,D,I,CX) const char * srepr(const T value);

NAMESPACE_END(miniten)

#endif // MINITEN__DISP_DECL
