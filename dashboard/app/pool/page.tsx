import { redirect } from "next/navigation";

/** The pool used to live here; it is the landing page now. Old links still work. */
export default function PoolRedirect() {
  redirect("/");
}
