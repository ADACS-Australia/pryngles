            """
            pattern=re.compile('(?:""")(.*?)(?:""")', re.DOTALL)
            strings=pattern.findall(block)
            docstring=""
            for string in strings:
                lines=string.split("\n+")
                for line in lines:
                    docstring+=f"\t{line}"
            """
