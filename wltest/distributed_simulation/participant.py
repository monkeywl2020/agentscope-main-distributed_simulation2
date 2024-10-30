# -*- coding: utf-8 -*-
"""A general dialog agent."""
import random
import time
import re
from typing import Optional, Union, Sequence
from types import GeneratorType
from typing import Optional, Generator, Tuple

from loguru import logger

from agentscope.message import Msg
from agentscope.agents import AgentBase

from textcontent import content_1200, content_2955,content_4900

_PREFIX_DICT = {}

class RandomParticipant(AgentBase):
    """A fake participant who generates number randomly."""

    def __init__(
        self,
        name: str,
        max_value: int = 100,
        sleep_time: float = 1.0,
    ) -> None:
        """Initialize the participant."""
        super().__init__(
            name=name,
        )
        self.max_value = max_value
        self.sleep_time = sleep_time

    def generate_random_response(self) -> str:
        """generate a random int"""
        time.sleep(self.sleep_time)
        return str(random.randint(0, self.max_value))

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        """Generate a random value"""
        # generate a response in content
        response = self.generate_random_response()
        msg = Msg(self.name, content=response, role="assistant")
        return msg


class LLMParticipant(AgentBase):
    """A participant agent who generates number using LLM."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        max_value: int = 1100,
    ) -> None:
        # 设置一个变量来记录访问大模型的初始时间
        self.startTime = 0
        # 设置一个变量来记录收到大模型首包时间
        self.firstPacketTime = 0
        # 设置一个变量来记录收完响应的时间
        self.endTime = 0

        # 记录首包的时间消耗
        self.firstPacketTimeCost = 0
        # 记录收完响应的时间消耗
        self.endTimeCost = 0

        """Initialize the participant."""
        print("LLMParticipant---------------------------------")
        super().__init__(
            name=name,
            model_config_name=model_config_name,
            use_memory=True,
        )
        self.max_value = max_value
        self.prompt = Msg(
            name="system",
            role="system",
            content=f"您正在参加一个游戏,你将提供一段文字。在游戏中每个人都会提供超过1000字的中文. 内容长度1000-{max_value}个左右,可以是文章或者小说内容。可以与历史信息相关或者不相关"
        )

    def parse_value(self, txt: str) -> str:
        """Parse the number from the response."""
        strlen = len(txt)
        if strlen == 0:
            logger.warning(
                f"Fail to parse value from [{txt}], use "
                f"{self.max_value // 2} instead.",
            )
            return str(self.max_value // 2)
        else:
            return strlen

    def log_msg(self, msg: Msg, disable_gradio: bool = False) -> None:
        """Print the message and save it into files. Note the message should be a
        Msg object."""

        if not isinstance(msg, Msg):
            raise TypeError(f"Get type {type(msg)}, expect Msg object.")

        print(msg.formatted_str(colored=True))

    def log_stream_msg(self, msg: Msg, last: bool = True) -> None:
        global _PREFIX_DICT

        # Print msg to terminal
        formatted_str = msg.formatted_str(colored=True)

        print_str = formatted_str[_PREFIX_DICT.get(msg.id, 0) :]

        if last:
            # Remove the prefix from the dictionary
            del _PREFIX_DICT[msg.id]

            #print(print_str)
        else:
            # Update the prefix in the dictionary
            _PREFIX_DICT[msg.id] = len(formatted_str)

            #print(print_str, end="")

    def speak(
        self,
        content: Union[str, Msg, Generator[Tuple[bool, str], None, None]],
    ) -> None:
        if isinstance(content, GeneratorType):
            # The streaming message must share the same id for displaying in
            # the agentscope studio.
            text_chunk = next(content)
            msg = Msg(name=self.name, content=text_chunk, role="assistant")

            self.firstPacketTime = time.time() # 记录收到大模型首包的时间
            self.firstPacketTimeCost = self.firstPacketTime - self.startTime # 记录收到首包的时间消耗
            
            for last, text_chunk in content:
                msg.content = text_chunk
                #self.log_stream_msg(msg, last=last)
        else:
            raise TypeError(
                "From version 0.0.5, the speak method only accepts str or Msg "
                f"object, got {type(content)} instead.",
            )

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        logger.info("============reply============")
        """Generate a value by LLM"""
        if self.memory:
            self.memory.add(x)

        # prepare prompt
        prompt = self.model.format(self.prompt, self.memory.get_memory())
        
        # 记录访问大模型的初始时间
        self.startTime = time.time()
        # call llm and generate response
        response = self.model(prompt)
        
        # 打印流式内容
        self.speak(response.stream)
        
        self.endTime = time.time() # 记录收完响应的时间
        self.endTimeCost = self.endTime - self.startTime # 记录收完响应的时间消耗

        #logger.info("------------------\n")
        #logger.info(response)
        print("---ccc---:",response,flush=True)
        print("---ccc222---frist packet cost:",self.firstPacketTimeCost ,flush=True)
        print("---ccc222---end packet cost:",self.endTimeCost,flush=True)

        #logger.info("------------------\n")
        response = self.parse_value(response.text)

        msg = Msg(self.name, response, role="assistant")

        # Record the message in memory
        if self.memory:
            self.memory.add(msg)

        return msg


class Moderator(AgentBase):
    """A Moderator to collect values from participants."""

    def __init__(
        self,
        name: str,
        part_configs: list[dict],
        agent_type: str = "random",
        max_value: int = 1100,
        sleep_time: float = 1.0,
    ) -> None:
        #print("Moderator---------------------------------")
        super().__init__(name)
        self.max_value = max_value
        if agent_type == "llm":
            self.participants = [
                LLMParticipant(
                    name=config["name"],
                    model_config_name=config["model_config_name"],
                    max_value=max_value,
                ).to_dist(
                    host=config["host"],
                    port=config["port"],
                )
                for config in part_configs
            ]
        else:
            self.participants = [
                RandomParticipant(
                    name=config["name"],
                    max_value=max_value,
                    sleep_time=sleep_time,
                ).to_dist(
                    host=config["host"],
                    port=config["port"],
                )
                for config in part_configs
            ]

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        results = []
        # 150 个字符长度的中文 content_4500 content_1200
        content = content_4900 + f"\n请作为参与游戏者生成一段中文文字,内容长度数量在 1000 and {self.max_value}之间.可以与上面的内容相关或者不相关。"
        #content = f"\n请作为参与游戏者生成一段中文文字,内容长度数量在 1000 and {self.max_value}之间.可以与上面的内容相关或者不相关。"
        msg = Msg(
            name="moderator",
            role="user",
            content= content
        )

        for p in self.participants:
            results.append(p(msg))
        summ = 0
        for r in results:
            try:
                summ += int(r.content)
            except Exception as e:
                print(e)
        return Msg(
            name=self.name,
            role="assistant",
            content={"sum": summ, "cnt": len(self.participants)},
        )
